from itertools import product


def get_configuration_weight(config):
    weights = []
    for i, c in enumerate(config):
        position_weight = len(config) - i
        if c == "2":
            type_weight = 3
        elif c in ["u", "d"]:
            type_weight = 2
        else:  # '0'
            type_weight = 1
        weights.append((position_weight, type_weight))
    return tuple(weights)


def generate_electron_configurations(
    num_electrons, num_orbitals, num_docc=0, num_active=None, save_docc=False
):
    """
    生成给定条件下的所有可能电子构型

    Args:
        num_electrons (int): 总电子数
        num_orbitals (int): 总轨道数
        num_docc (int): 双占据轨道数（这些轨道固定为双占据，不参与激发）
        num_active (int): 活性轨道数（参与激发的轨道数）
    """
    if num_active is None:
        num_active = num_orbitals - num_docc

    if num_active > num_orbitals - num_docc:
        raise ValueError("活性轨道数不能大于可用轨道数")

    # 计算活性空间中的电子数
    active_electrons = num_electrons - 2 * num_docc

    def is_valid_config(config):
        # 检查双占据部分
        if not all(c == "2" for c in config[:num_docc]):
            return False

        # 检查非活性部分
        if not all(c == "0" for c in config[num_docc + num_active :]):
            return False

        # 检查活性空间中的电子数
        active_part = config[num_docc : num_docc + num_active]
        active_e = sum(
            2 if x == "2" else 1 if x in ["u", "d"] else 0 for x in active_part
        )
        if active_e != active_electrons:
            return False

        # 检查自旋平衡（alpha = beta）
        # 双占据的轨道已经是平衡的，只需检查活性空间
        alpha_count = sum(1 for x in active_part if x == "u")
        beta_count = sum(1 for x in active_part if x == "d")
        # 注意：双占据的不需要计算因为它们已经平衡
        return alpha_count == beta_count

    def generate_base_configs():
        # 生成固定的双占据部分
        docc_part = "2" * num_docc

        # 生成非活性部分
        inactive_part = "0" * (num_orbitals - num_docc - num_active)

        # 生成活性空间的所有可能构型
        electrons = ["2", "u", "d", "0"]
        active_configs = ["".join(p) for p in product(electrons, repeat=num_active)]

        # 组合所有部分
        return [
            docc_part + active_config + inactive_part
            for active_config in active_configs
        ]

    base_configs = generate_base_configs()
    valid_configs = [c for c in base_configs if is_valid_config(c)]
    tmp = sorted(valid_configs, key=get_configuration_weight, reverse=True)
    if save_docc:
        return tmp
    else:
        return [x[num_docc:] for x in tmp]


def save_configurations_to_file(configs, filename="electron_configurations.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for config in configs:
            f.write(f"{config}\n")


def main():
    print("Please enter electron configuration parameters:")
    try:
        num_electrons = int(input("Total number of electrons: "))
        num_orbitals = int(input("Total number of orbitals: "))
        num_docc = int(input("Number of doubly occupied orbitals: "))
        num_active = int(input("Number of active orbitals: "))
        save_docc = input("Save doubly occupied orbitals? (y/n): ")

        # Print summary before generation
        print("\nConfiguration Summary:")
        print(f"Total electrons: {num_electrons}")
        print(f"Electrons in doubly occupied orbitals: {2 * num_docc}")
        print(f"Electrons in active space: {num_electrons - 2 * num_docc}")
        print(f"Active orbitals: {num_active}")
        print(f"Save doubly occupied orbitals:{save_docc}")

        if save_docc.lower() == "y":
            configs = generate_electron_configurations(
                num_electrons,
                num_orbitals,
                num_docc=num_docc,
                num_active=num_active,
                save_docc=True,
            )
        else:
            configs = generate_electron_configurations(
                num_electrons,
                num_orbitals,
                num_docc=num_docc,
                num_active=num_active,
                save_docc=False,
            )
        print(f"\nGenerated {len(configs)} electron configurations:")
        # for config in configs:
        #    print(config)

        save = input("\nSave to file? (y/n): ")
        if save.lower() == "y":
            filename = input(
                "Enter filename (default: electron_configurations.txt): "
            ).strip()
            if not filename:
                filename = "electron_configurations.txt"

            save_configurations_to_file(configs, filename)
            print(f"Configurations saved to {filename}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure all inputs are valid integers")


if __name__ == "__main__":
    main()
