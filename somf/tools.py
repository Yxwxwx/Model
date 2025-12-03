import numpy as np


def dump_cpx_FCIDUMP(
    filename,
    n_sites,
    n_elec,
    twos,
    isym,
    orb_sym,
    const_e,
    h1e,
    g2e,
    tol=1e-13,
    pg="c1",
):
    """
    Write FCI options and integrals to FCIDUMP file.
    Args:
        filename : str
        tol : threshold for terms written into file
    """
    assert not isinstance(h1e, tuple)
    with open(filename, "w") as fout:
        fout.write(f" &FCI NORB={n_sites:4d},NELEC={n_elec:4d},MS2={twos},\n")
        # NO pg
        fout.write("  ORBSYM=%s\n" % ("1," * n_sites))
        fout.write("  ISYM=1,\n")
        fout.write(" &END\n")
        output_format_cpx = "%20.16f%20.16f%4d%4d%4d%4d\n"

        nmo = n_sites

        def write_eri(fout, eri):
            assert eri.size == nmo**4
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            if abs(eri[i, j, k, l]) > tol:
                                fout.write(
                                    output_format_cpx
                                    % (
                                        np.real(eri[i, j, k, l]),
                                        np.imag(eri[i, j, k, l]),
                                        i + 1,
                                        j + 1,
                                        k + 1,
                                        l + 1,
                                    )
                                )

        def write_h1e(fout, h1e):
            h = h1e.reshape(nmo, nmo)
            for i in range(nmo):
                for j in range(0, i + 1):
                    if abs(h[i, j]) > tol:
                        fout.write(
                            output_format_cpx
                            % (np.real(h[i, j]), np.imag(h[i, j]), i + 1, j + 1, 0, 0)
                        )

        write_eri(fout, g2e)
        write_h1e(fout, h1e)
        fout.write(output_format_cpx % (np.real(const_e), np.imag(const_e), 0, 0, 0, 0))
