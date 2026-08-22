import os
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
EnergyFunction = Callable[[float, float, float], float]


_ENERGY_FUNCTION: EnergyFunction | None = None


def _init_worker(
    energy_function: EnergyFunction,
) -> None:
    global _ENERGY_FUNCTION
    _ENERGY_FUNCTION = energy_function


def _evaluate_point(
    task: tuple[
        int,
        int,
        int,
        float,
        float,
        float,
    ],
) -> tuple[int, int, int, float]:
    if _ENERGY_FUNCTION is None:
        raise RuntimeError("Worker energy function is not initialized.")

    ix, iy, iz, x, y, z = task

    energy = _ENERGY_FUNCTION(
        x,
        y,
        z,
    )

    return ix, iy, iz, float(energy)


def _save_checkpoint(
    filename: Path,
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    pes: FloatArray,
) -> None:
    """
    Atomically save PES checkpoint.
    """
    tmp = filename.with_suffix(".tmp.npz")

    np.savez(
        tmp,
        x=x,
        y=y,
        z=z,
        pes=pes,
    )

    tmp.replace(filename)


def _load_checkpoint(
    filename: Path,
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
) -> FloatArray:
    shape = (
        x.size,
        y.size,
        z.size,
    )

    if not filename.exists():
        return np.full(
            shape,
            np.nan,
            dtype=float,
        )

    with np.load(filename) as data:
        old_x = data["x"]
        old_y = data["y"]
        old_z = data["z"]
        pes = np.asarray(
            data["pes"],
            dtype=float,
        )

    if not (
        np.array_equal(old_x, x)
        and np.array_equal(old_y, y)
        and np.array_equal(old_z, z)
    ):
        raise ValueError("Checkpoint grid does not match current grid.")

    if pes.shape != shape:
        raise ValueError("Checkpoint PES shape does not match current grid.")

    return pes


def _iter_tasks(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    pes: FloatArray,
) -> Iterator[
    tuple[
        int,
        int,
        int,
        float,
        float,
        float,
    ]
]:
    """
    Yield unfinished PES grid points.
    """
    for ix, xi in enumerate(x):
        for iy, yi in enumerate(y):
            for iz, zi in enumerate(z):
                if np.isfinite(pes[ix, iy, iz]):
                    continue

                yield (
                    ix,
                    iy,
                    iz,
                    float(xi),
                    float(yi),
                    float(zi),
                )


def _get_nworkers(
    nworkers: int | None,
) -> int:
    """
    Determine number of worker processes.

    Priority:
        explicit nworkers
        -> SLURM_CPUS_PER_TASK
        -> 1
    """
    if nworkers is not None:
        if nworkers <= 0:
            raise ValueError("nworkers must be positive.")

        return nworkers

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is not None:
        return max(
            1,
            int(slurm_cpus),
        )

    # Avoid accidentally using all CPUs on a login node.
    return 1


def build_pes_3d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
    energy_function: EnergyFunction,
    checkpoint: str | Path | None = None,
    checkpoint_interval: int = 100,
    verbose: bool = True,
    nworkers: int | None = None,
) -> FloatArray:
    """
    Build a 3D potential-energy surface.

    The supplied energy function must have the form

        energy_function(x, y, z) -> energy

    The electronic-structure backend is completely
    user-defined.

    Parallel execution is performed with independent
    worker processes. If nworkers is not explicitly
    specified, SLURM_CPUS_PER_TASK is used when running
    inside Slurm.

    Parameters
    ----------
    x, y, z
        Cartesian grids.

    energy_function
        User-supplied energy function.

    checkpoint
        Optional restartable checkpoint.

    checkpoint_interval
        Save checkpoint after this many newly completed
        grid points.

    verbose
        Print progress information.

    nworkers
        Number of worker processes. If None, use
        SLURM_CPUS_PER_TASK or fall back to 1.

    Returns
    -------
    pes
        Potential-energy surface with shape
        (nx, ny, nz).
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    z = np.asarray(
        z,
        dtype=float,
    )

    checkpoint_path = Path(checkpoint) if checkpoint is not None else None

    if checkpoint_path is None:
        pes = np.full(
            (
                x.size,
                y.size,
                z.size,
            ),
            np.nan,
            dtype=float,
        )
    else:
        pes = _load_checkpoint(
            checkpoint_path,
            x,
            y,
            z,
        )

    total = pes.size

    finished = int(np.count_nonzero(np.isfinite(pes)))

    workers = _get_nworkers(nworkers)

    if verbose:
        print(f"PES: {finished}/{total} points finished.")

        print(f"PES workers: {workers}")

    if finished == total:
        return pes

    tasks = _iter_tasks(
        x,
        y,
        z,
        pes,
    )

    # --------------------------------------------------------
    # Serial execution
    # --------------------------------------------------------

    if workers == 1:
        for task in tasks:
            ix, iy, iz, xi, yi, zi = task

            energy = energy_function(
                xi,
                yi,
                zi,
            )

            pes[
                ix,
                iy,
                iz,
            ] = energy

            finished += 1

            if verbose and finished % checkpoint_interval == 0:
                print(
                    f"PES: {finished}/{total}",
                    flush=True,
                )

            if checkpoint_path is not None and finished % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    x,
                    y,
                    z,
                    pes,
                )

    # --------------------------------------------------------
    # Multiprocessing execution
    # --------------------------------------------------------

    else:
        #
        # Do not submit all ~262000 tasks simultaneously.
        # Keep only a small number of outstanding futures.
        #
        max_pending = 4 * workers

        # spawn is safer than fork for numerical libraries
        # using BLAS/OpenMP.
        context = get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(energy_function,),
        ) as executor:
            pending = set()

            #
            # Initial task queue.
            #
            for _ in range(max_pending):
                try:
                    task = next(tasks)
                except StopIteration:
                    break

                pending.add(
                    executor.submit(
                        _evaluate_point,
                        task,
                    )
                )

            #
            # Keep queue filled while tasks finish.
            #
            while pending:
                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    ix, iy, iz, energy = future.result()

                    pes[
                        ix,
                        iy,
                        iz,
                    ] = energy

                    finished += 1

                    if verbose and finished % checkpoint_interval == 0:
                        print(
                            f"PES: {finished}/{total}",
                            flush=True,
                        )

                    if (
                        checkpoint_path is not None
                        and finished % checkpoint_interval == 0
                    ):
                        _save_checkpoint(
                            checkpoint_path,
                            x,
                            y,
                            z,
                            pes,
                        )

                    try:
                        task = next(tasks)
                    except StopIteration:
                        continue

                    pending.add(
                        executor.submit(
                            _evaluate_point,
                            task,
                        )
                    )

    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path,
            x,
            y,
            z,
            pes,
        )

    if verbose:
        print(f"PES: {total}/{total} points finished.")

    return pes
