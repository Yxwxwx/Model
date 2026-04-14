import numpy as np


def get_random_restricted_mps(
    self,
    tag,
    bond_dim=500,
    center=0,
    dot=2,
    target=None,
    nroots=1,
    occs=None,
    full_fci=False,
    left_vacuum=None,
    nrank_mrci=0,
    orig_dot=False,
    n_hole=None,
    n_core=None,
    n_inact=None,
    n_exter=None,
    n_act=None,
    core_sz=None,  # 【新增】限制核心区域的 2*Sz (例如 1 代表 alpha, -1 代表 beta)
):
    bw = self.bw
    if target is None:
        target = self.target
    if left_vacuum is None:
        left_vacuum = self.left_vacuum

    # 1. 初始化 MPSInfo
    if nroots == 1:
        mps_info = bw.brs.MPSInfo(self.n_sites, self.vacuum, target, self.ghamil.basis)
        mps = bw.bs.MPS(self.n_sites, center, dot if orig_dot else 1)
    else:
        targets = bw.VectorSX([target]) if isinstance(target, bw.SX) else target
        mps_info = bw.brs.MultiMPSInfo(
            self.n_sites, self.vacuum, targets, self.ghamil.basis
        )
        mps = bw.bs.MultiMPS(self.n_sites, center, dot if orig_dot else 1, nroots)

    mps_info.tag = tag
    if full_fci:
        mps_info.set_bond_dimension_full_fci(left_vacuum, self.vacuum)
    else:
        mps_info.set_bond_dimension_fci(left_vacuum, self.vacuum)

    # 2. RAS 限制 (引入 Sz)
    if n_hole is not None and n_core is not None:
        nelec_total = target.n  # 目标总电子数
        target_sz = target.twos  # 目标总 2*Sz

        nc = n_core
        ni = n_inact if n_inact is not None else 0
        ne = n_exter if n_exter is not None else 0
        na = n_act if n_act is not None else (self.n_sites - nc - ni - ne)

        # 【修改】同时接收 n 和 sz 进行判断
        def restrict_qn(i, dims_fci, conds):
            dims = dims_fci[i]
            for cond in conds:
                for q in range(dims.n):
                    # 提取该维度的 n 和 sz 量子数
                    q_n = dims.quanta[q].n
                    q_sz = dims.quanta[q].twos
                    if cond(q_n, q_sz):
                        dims.n_states[q] = 0

        kp = np.array([0, nc, nc + ni, nc + ni + ne, nc + ni + ne + na])

        core_elec = 2 * nc - n_hole
        inact_elec = core_elec + 2 * ni

        nminp = np.array(
            [
                0,
                core_elec,
                inact_elec - nrank_mrci,
                inact_elec - nrank_mrci,
                nelec_total,
            ]
        )
        nmaxp = np.array(
            [0, core_elec, inact_elec, inact_elec + nrank_mrci, nelec_total]
        )

        for i in range(self.n_sites + 1):
            idx_list = np.argwhere(kp >= i).flatten()
            idx = idx_list[0] - 1 if idx_list.size > 0 else 0
            idx = max(0, idx)
            ip = i - kp[idx]

            # 左侧必须达到的最小/最大电子数
            min_idx = max(nminp[idx], nminp[idx + 1] - 2 * (kp[idx + 1] - kp[idx] - ip))
            max_idx = min(nmaxp[idx + 1], nmaxp[idx] + 2 * ip)

            # --- 处理 Left Block ---
            def left_cond(n, sz):
                # 电子数不达标，剔除
                if n < min_idx or n > max_idx:
                    return True
                # 【新增】核心区域边界处的自旋过滤
                if core_sz is not None and i == nc:
                    if n == core_elec and sz != core_sz:
                        return True
                return False

            restrict_qn(i, mps_info.left_dims_fci, [left_cond])

            # --- 处理 Right Block ---
            min_right = nelec_total - max_idx
            max_right = nelec_total - min_idx

            def right_cond(n, sz):
                if n < min_right or n > max_right:
                    return True
                # 【新增】右侧匹配左侧自旋: Sz_right = Target_Sz - Sz_left
                if core_sz is not None and i == nc:
                    if n == (nelec_total - core_elec) and sz != (target_sz - core_sz):
                        return True
                return False

            restrict_qn(i, mps_info.right_dims_fci, [right_cond])

        for ldf in mps_info.left_dims_fci:
            ldf.collect()
        for rdf in mps_info.right_dims_fci:
            rdf.collect()

    # 3. 初始化与截断
    if occs is not None:
        mps_info.set_bond_dimension_using_occ(bond_dim, bw.b.VectorDouble(occs))
    else:
        mps_info.set_bond_dimension(bond_dim)
    mps_info.bond_dim = bond_dim
    mps.initialize(mps_info)
    mps.random_canonicalize()

    if nroots == 1:
        mps.tensors[mps.center].normalize()
    else:
        for xwfn in mps.wfns:
            xwfn.normalize()

    mps.save_mutable()
    mps_info.save_mutable()
    mps.save_data()
    mps_info.save_data(self.scratch + "/%s-mps_info.bin" % tag)

    if dot != 1 and not orig_dot:
        mps = self.adjust_mps(mps, dot=dot)[0]
    return mps


if __name__ == "__main__":
    from pyscf import gto, scf
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    DMRGDriver.get_random_restricted_mps = get_random_restricted_mps
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8

    # 注意：XPS初态(N-1)的 charge 应该是 +1，如果是 Auger 末态才是 +2。这里我先帮你改回 XPS 的设定
    mol = gto.M(atom="Ne 0 0 0", basis="unc-aug-cc-pVTZ", charge=+1, spin=1)
    mf = scf.ROHF(mol).run()

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
        mf, ncore=0, ncas=5, g2e_symm=1
    )

    print(f"ncas = {ncas}, n_elec = {n_elec}, spin = {spin}")

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)

    # 制备 XPS 初态
    ket = driver.get_random_restricted_mps(
        tag="GS",
        bond_dim=250,
        nroots=1,
        n_hole=1,
        n_core=1,
        n_inact=0,
        n_exter=0,
        n_act=4,
        # core_sz=1,  # 【关键设定】=1 代表核心保留 1 个 alpha 电子；=-1 代表保留 beta 电子
    )

    energy = driver.dmrg(
        mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1
    )
    print("DMRG energy (Ne+ Core-Hole State) = %20.15f" % energy)

    # 打印对角线验证
    rdm1 = driver.get_1pdm(ket)
    rdm1 = rdm1[0] + rdm1[1]
    print("总电子占据:", rdm1.diagonal())
