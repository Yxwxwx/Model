import numpy as np
import time


class MPO:
    """
    build MPO list for DMRG.
    :param single_mpo: a numpy ndarray with ndim=4.
    The first 2 dimensions reprsents the square shape of the MPO and the last 2 dimensions are physical dimensions.
    :param site_num: the total number of sites
    :param regularize: whether regularize the mpo so that it represents the average over all sites.
    :return MPO list
    The bond order of the local operator is [left, right, up, down], as
                                    | 2
                                0 --#-- 1
                                    | 3
    """

    def __init__(self, single_mpo, site_num):
        # the first MPO, only contains the last row
        mpo_first = single_mpo[-1].copy()
        mpo_first = mpo_first.reshape((1,) + mpo_first.shape)
        # the last MPO, only contains the first column
        mpo_last = single_mpo[:, 0].copy()
        mpo_last = mpo_last.reshape((mpo_last.shape[0],) + (1,) + mpo_last.shape[1:])
        self.mpo_list = (
            [mpo_first] + [single_mpo.copy() for _ in range(site_num - 2)] + [mpo_last]
        )

    @property
    def mpo(self):
        return self.mpo_list


class MPS:
    """
    build MPS list for DMRG.
    local matrix product state tensor
    bond order: [left, physical, right]
                    1
                    |
                0 --*-- 2
    """

    def __init__(self, phy_dim, bond_dim, site_num, error_thresh=0.0):
        mps_first = np.random.random((1, phy_dim, bond_dim))
        mps_last = np.random.random((bond_dim, phy_dim, 1))
        self.mps_list = (
            [mps_first]
            + [
                np.random.random((bond_dim, phy_dim, bond_dim))
                for _ in range(site_num - 2)
            ]
            + [mps_last]
        )

    @property
    def mps(self):
        return self.mps_list


def svd_compress(tensor, direction, maxM):
    """
    Perform svd compression on the self.matrix. Used in the canonical process.
    :param direction: To which the matrix is compressed
    :return: The u,s,v value of the svd decomposition. Truncated if self.thresh is provided.
    """
    left_argument_set = ["l", "left"]
    right_argument_set = ["r", "right"]
    assert direction in (left_argument_set + right_argument_set)
    left, phy_dim, right = tensor.shape
    if direction in left_argument_set:
        u, s, v = np.linalg.svd(
            tensor.reshape(left * phy_dim, right), full_matrices=False
        )
    else:
        u, s, v = np.linalg.svd(
            tensor.reshape(left, phy_dim * right), full_matrices=False
        )
    if len(s) > maxM:  # do truncation
        return u[:, :maxM], s[:maxM], v[:maxM, :]
    return u, s, v


# def svd_compress(tensor, direction, maxM):
#     """
#     Perform SVD compression with truncated singular values using sparse SVD for efficiency.
#     """
#     left_args = ["l", "left"]
#     right_args = ["r", "right"]
#     assert direction in (left_args + right_args)

#     left, phy_dim, right = tensor.shape
#     # Reshape tensor based on direction
#     if direction in left_args:
#         mat = tensor.reshape(left * phy_dim, right)
#     else:
#         mat = tensor.reshape(left, phy_dim * right)

#     m, n = mat.shape
#     min_dim = min(m, n)

#     if maxM < min_dim:
#         # Use sparse SVD with truncation
#         k = min(maxM, min_dim - 1)  # Ensure k < min_dim
#         if k <= 0:
#             u, s, vh = np.linalg.svd(mat, full_matrices=False)
#         else:
#             u, s, v = svds(mat, k=k, which="LM")
#             # svds returns s in ascending order, reverse for consistency
#             s = s[::-1]
#             u = u[:, ::-1]
#             v = v[:, ::-1]
#             vh = v.T.conj()  # Convert to row vectors
#     else:
#         # Fallback to full SVD if maxM >= min_dim
#         u, s, vh = np.linalg.svd(mat, full_matrices=False)

#     # Truncate if necessary (mainly for full SVD case)
#     if len(s) > maxM:
#         u = u[:, :maxM]
#         s = s[:maxM]
#         vh = vh[:maxM, :]

#     return u, s, vh


class DMRG:
    """
    DMRG algorithm for MPS.
    """

    def __init__(
        self, mpo, mps, max_bond_dimension=0, max_sweeps=20, error_threshold=1e-6
    ):
        """
        Initialize a MatrixProductState with given bond dimension.
        :param mpo_list: the list for MPOs. The site num depends on the length of the list
        :param max_bond_dimension: the bond dimension required. The higher bond dimension, the higher accuracy and compuational cost
        :param error_threshold: error threshold used in svd compressing of the matrix state.
        The lower the threshold, the higher the accuracy.
        """

        if max_bond_dimension == 0:
            raise ValueError("Must provide max_bond_dimension")
        self.max_bond_dimension = max_bond_dimension
        self.max_sweeps = max_sweeps
        self.error_threshold = error_threshold
        self.site_num = len(mpo)
        # dummy local tensor and local operator
        self.mpo_list = [np.zeros((0, 0, 0))] + mpo + [np.zeros((0, 0, 0))]
        self.mps_list = [np.zeros((0, 0, 0))] + mps + [np.zeros((0, 0, 0))]
        self.F_list = (
            [np.ones((1,) * 6)]
            + [None for _ in range(self.site_num)]
            + [np.ones((1,) * 6)]
        )
        self.L_list = self.F_list.copy()
        self.R_list = self.F_list.copy()
        # do right canonicalization
        self.right_canonicalize_from(idx=self.site_num)

    def update_local_site(self, idx, newState):
        self.mps_list[idx] = newState
        # since the current site has changed,
        # tensor F at the current site need to be updated.
        self.F_list[idx] = None
        # the current site has influence on the tensor L of all the sites on its right
        for i in range(idx + 1, self.site_num + 1):
            if self.L_list[i] is None:
                break
            self.L_list[i] = None
        # on the tensor R of all the sites on its left
        for i in range(idx - 1, 0, -1):
            if self.R_list[i] is None:
                break
            self.R_list[i] = None

    # head dummy site: idx == 0
    # real sites : idx == 1~size
    # last dummy site: idx == size+1
    def left_canonicalize_at(self, idx):
        if idx >= self.site_num:
            return
        site = self.mps_list[idx]
        left, phy_dim, right = site.shape
        u, s, v = svd_compress(site, "left", self.max_bond_dimension)
        # update the MPS at idx
        self.update_local_site(idx, newState=u.reshape((left, phy_dim, -1)))
        # the next site is on the right of the current site
        self.update_local_site(
            idx + 1,
            newState=np.tensordot(
                np.dot(np.diag(s), v), self.mps_list[idx + 1], axes=[1, 0]
            ),
        )

    def left_canonicalize_from(self, idx):
        for i in range(idx, self.site_num):
            self.left_canonicalize_at(i)

    # head dummy site: idx == 0
    # real sites : idx == 1~size
    # last dummy site: idx == size+1
    def right_canonicalize_at(self, idx):
        if idx <= 1:
            return
        site = self.mps_list[idx]
        left, phy_dim, right = site.shape
        u, s, v = svd_compress(site, "right", self.max_bond_dimension)
        # update the MPS at idx
        self.update_local_site(idx, newState=v.reshape((-1, phy_dim, right)))
        # the next site is on the left of the current site
        self.update_local_site(
            idx - 1,
            np.tensordot(
                self.mps_list[idx - 1].data, np.dot(u, np.diag(s)), axes=[2, 0]
            ),
        )

    def right_canonicalize_from(self, idx):
        for i in range(idx, 1, -1):
            self.right_canonicalize_at(i)

    def tensorF_at(self, idx):
        """
        calculate F for this site.
        graphical representation (* for MPS and # for MPO,
        numbers represents a set of imaginary bond dimensions used for comments below):
                                  1 --*-- 5
                                      | 4
                                  2 --#-- 3
                                      | 4
                                  1 --*-- 5
        :return the calculated F
        """
        if self.F_list[idx] is None:
            # compute tensor F for idx
            site = self.mps_list[idx]
            operator = self.mpo_list[idx]
            # site is (1,4,5)
            # operator is (2,3,4,4)
            # compute <site|operator
            # contract 4, F is (1,5,2,3,4)
            F = np.tensordot(site.conj(), operator, axes=[1, 2])
            # compute <site|operator|site>
            # contract 4
            F = np.tensordot(F, site, axes=[4, 1])
            # F is x
            self.F_list[idx] = F
        return self.F_list[idx]

    def tensorL_at(self, idx):
        """
        calculate L in a recursive way
        """
        if self.L_list[idx] is None:
            if idx <= 1:  # head dummy site
                self.L_list[idx] = self.tensorF_at(idx)
            else:
                leftL = self.tensorL_at(idx - 1)
                currentF = self.tensorF_at(idx)
                """
                do the contraction. 
                graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
                  0 --*-- 1          0 --*-- 1                   0 --*-- 2                     0 --*-- 1          
                      |                  |                           |                             |     
                  2 --#-- 3     +    2 --#-- 3  --tensordot-->   4 --#-- 1    --transpose-->   2 --#-- 3                
                      |                  |                           |                             |     
                  4 --*-- 5          4 --*-- 5                   3 --*-- 5                     4 --*-- 5          

                """
                # tensordot (0,2,4,1,3,5) -transpose-> (0,1,2,3,4,5)
                currentL = np.tensordot(
                    leftL, currentF, axes=[[1, 3, 5], [0, 2, 4]]
                ).transpose((0, 3, 1, 4, 2, 5))
                self.L_list[idx] = currentL
        return self.L_list[idx]

    def tensorR_at(self, idx):
        """
        calculate R in a recursive way
        """
        if self.R_list[idx] is None:
            if idx >= self.site_num:
                self.R_list[idx] = self.tensorF_at(idx)
            else:
                rightR = self.tensorR_at(idx + 1)
                currentF = self.tensorF_at(idx)
                currentR = np.tensordot(
                    currentF, rightR, axes=[[1, 3, 5], [0, 2, 4]]
                ).transpose((0, 3, 1, 4, 2, 5))
                self.R_list[idx] = currentR
        return self.R_list[idx]

    def variational_tensor_at(self, idx):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        left, phy_dim, right = self.mps_list[idx].shape
        operator = self.mpo_list[idx]
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1                
              |                | 2                         |    | 6      
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5                 
              |                | 3                         |    | 7      
          4 --*-- 5                                    3 --*-- 4                
              L                MPO                       left_middle
        """
        result = np.tensordot(self.tensorL_at(idx - 1), operator, axes=[3, 0])
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 8 --*-- 9      
              |    | 6              |                           |    | 6  |   
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 10       
              |    | 7              |                           |    | 7  |   
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 11--*-- 12 
            left_middle             R                       raw variational tensor
        Note the dimension of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        result = np.tensordot(result, self.tensorR_at(idx + 1), axes=[5, 2])
        result = result.reshape(  # (1,4,6,7,8,11)
            left,
            left,
            phy_dim,
            phy_dim,
            right,
            right,
        ).transpose((0, 2, 4, 1, 3, 5))  # (1,6,8,4,7,11)
        return result

    def sweep_at(self, idx, direction="left"):
        """
        DMRG sweep
        """
        left_argument_set = ["l", "left"]
        right_argument_set = ["r", "right"]

        assert direction in (left_argument_set + right_argument_set)
        left, phy_dim, right = self.mps_list[idx].shape
        localDimension = left * phy_dim * right
        # reshape the variational tensor to a square matrix
        variationalTensor = self.variational_tensor_at(idx).reshape(
            (localDimension, localDimension)
        )
        # solve for eigen values and vectors
        eigVal, eigVec = np.linalg.eigh(variationalTensor)
        # update current site
        self.update_local_site(
            idx, newState=eigVec[:, 0].reshape((left, phy_dim, right))
        )
        # normalization
        if direction in left_argument_set:
            self.left_canonicalize_at(idx)
        else:
            self.right_canonicalize_at(idx)
        return eigVal[0]  # ground state energy

    def kernel(self):
        """
        The main kernel for DMRG algorithm.
        """
        E_old = 0.0
        print("Max bond dimension:", self.max_bond_dimension)
        print("Max sweeps:", self.max_sweeps)
        print("Error threshold:", self.error_threshold)
        print("********* naive DMRG for spin model *********")
        for sweep in range(self.max_sweeps):
            print(f"DMRG sweep {sweep + 1}/{self.max_sweeps}")
            print(">>>>>>>>>> sweep from left to right >>>>>>>>>>")
            # left -> right sweep
            for idx in range(1, self.site_num + 1):
                E_new = self.sweep_at(idx, "left")
                print(f"Left sweep at site {idx}, energy: {E_new:.6f}")
            # do right sweep
            for idx in range(self.site_num, 0, -1):
                E_new = self.sweep_at(idx, "right")
                print(f"Right sweep at site {idx}, energy: {E_new:.6f}")
            # check convergence
            if abs(E_new - E_old) < self.error_threshold:
                print(f"DMRG converged in {sweep + 1} sweeps.")
                print(f"Final energy: {E_new:.6f}")
                return E_new
            E_old = E_new
        print(f"DMRG did not converge after {self.max_sweeps} sweeps.")
        print(f"Final energy: {E_new:.6f}")


if __name__ == "__main__":
    MAX_BOND_DIMENSION = 50
    MAX_SWEEPS = 20
    PHY_DIM = 2
    SITE_NUM = 10
    ERROR_THRESHOLD = 1e-7
    # Define local operators
    identity = np.identity(2)
    zeros = np.zeros((2, 2))
    sz = np.array([[0.5, 0], [0, -0.5]])
    sp = np.array([[0, 0], [1, 0]])  # S^+
    sm = np.array([[0, 1], [0, 0]])  # S^-
    # MPO tensors for each site in the chain
    w_bulk = np.array(
        [
            [identity, zeros, zeros, zeros, zeros],
            [sp, zeros, zeros, zeros, zeros],
            [sm, zeros, zeros, zeros, zeros],
            [sz, zeros, zeros, zeros, zeros],
            [zeros, 0.5 * sm, 0.5 * sp, sz, identity],
        ]
    )
    mpo = MPO(w_bulk, SITE_NUM)
    mps = MPS(PHY_DIM, MAX_BOND_DIMENSION, SITE_NUM, ERROR_THRESHOLD)
    dmrg = DMRG(
        mpo=mpo.mpo,
        mps=mps.mps,
        max_bond_dimension=MAX_BOND_DIMENSION,
        max_sweeps=MAX_SWEEPS,
        error_threshold=ERROR_THRESHOLD,
    )
    start = time.time()
    dmrg.kernel()
    print("Time cost:", time.time() - start, "s")
