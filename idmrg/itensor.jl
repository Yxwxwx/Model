using ITensors, ITensorMPS
let
  N = 100
  sites = siteinds("S=1/2", N)
  os = OpSum()
  for j = 1:N-1
    os += "Sz", j, "Sz", j + 1
    os += 1 / 2, "S+", j, "S-", j + 1
    os += 1 / 2, "S-", j, "S+", j + 1
  end
  H = MPO(os, sites)

  psi0 = random_mps(sites, linkdims=10)

  nsweeps = 8
  maxdim = 10
  cutoff = [1E-10]

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  return energy / N
end