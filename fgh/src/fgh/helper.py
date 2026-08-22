from pyscf.lib import logger


def second_order_scf(mf):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ

    mf = mf.newton()
    mf.conv_tol = 1e-12
    mf.max_cycle = 100

    mf.kernel(
        mo_coeff,
        mo_occ,
    )

    return mf


def stable_opt_internal(mf):
    log = logger.new_logger(mf)

    mo1, _, stable, _ = mf.stability(
        internal=True,
        external=False,
        return_status=True,
    )

    cyc = 0

    while not stable and cyc < 10:
        log.note(
            "Try to optimize orbitals until stable, attempt %d",
            cyc + 1,
        )

        dm1 = mf.make_rdm1(
            mo1,
            mf.mo_occ,
        )

        mf.kernel(dm0=dm1)

        # If the stability restart causes ordinary SCF
        # convergence problems, switch to Newton.
        if not mf.converged:
            log.note(
                "SCF did not converge after stability rotation; "
                "switching to Newton solver."
            )

            mf = mf.newton()
            mf.conv_tol = 1e-12
            mf.max_cycle = 100

            mf.kernel(dm0=dm1)

        if not mf.converged:
            raise RuntimeError("SCF failed during internal stability optimization.")

        mo1, _, stable, _ = mf.stability(
            internal=True,
            external=False,
            return_status=True,
        )

        cyc += 1

    if stable:
        log.note("RHF wave function is internally stable.")
    else:
        log.warn(
            "Internal stability optimization failed after %d attempts.",
            cyc,
        )

    return mf
