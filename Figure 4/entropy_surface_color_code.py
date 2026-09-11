import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as line_2d

plt.style.use("rgplot")

p_list = np.logspace(-2, -0.3, 10)
threshold = 0.1094


color_code_stabilizer_supports = {
    5: (
        (0, 1, 2, 3),
        (1, 3, 5, 6),
        (2, 3, 4, 5, 7, 8),
        (4, 7, 10, 11),
        (5, 6, 8, 9, 12, 13),
        (10, 11, 14, 15),
        (7, 8, 11, 12, 15, 16),
        (12, 13, 16, 17),
        (9, 13, 17, 18),
    ),
    7: (
        (0, 1, 2, 3),
        (1, 3, 5, 6),
        (2, 3, 4, 5, 7, 8),
        (4, 7, 10, 11),
        (5, 6, 8, 9, 12, 13),
        (7, 8, 11, 12, 15, 16),
        (9, 13, 17, 18),
        (10, 11, 14, 15, 19, 20),
        (12, 13, 16, 17, 21, 22),
        (14, 19, 24, 25),
        (15, 16, 20, 21, 26, 27),
        (17, 18, 22, 23, 28, 29),
        (19, 20, 25, 26, 31, 32),
        (21, 22, 27, 28, 33, 34),
        (23, 29, 35, 36),
        (24, 25, 30, 31),
        (26, 27, 32, 33),
        (28, 29, 34, 35),
    ),
}

color_code_logical_supports = {
    5: (0, 2, 4, 10, 14),
    7: (0, 2, 4, 10, 14, 24, 30),
}


# Compute the binary Shannon entropy in bits.
def _binary_entropy(probability):
    probability = np.asarray(probability, dtype=np.float64)
    entropy = np.zeros_like(probability)
    valid = (probability > 0.0) & (probability < 1.0)
    p = probability[valid]
    entropy[valid] = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return entropy


# Compute the entropy H(E) of independent physical errors.
def H_E(check_matrix, probability):
    return check_matrix.shape[1] * _binary_entropy(probability)


# Build the surface-code check matrix and logical support.
def build_surface_code(distance):
    half_distance = (distance - 1) // 2
    check_matrix = np.zeros(
        ((distance**2 - 1) // 2, distance**2),
        dtype=np.uint8,
    )

    for index in range(half_distance):
        check_matrix[index, (2 * index, 2 * index + 1)] = 1

    for row in range(distance - 1):
        for column in range(half_distance):
            check_index = (row + 1) * half_distance + column
            offset = 1 if row % 2 == 0 else 0
            left = row * distance + 2 * column + offset
            support = (
                left,
                left + 1,
                left + distance,
                left + distance + 1,
            )
            check_matrix[check_index, support] = 1

    bottom_row = distance * (distance - 1)
    for index in range(half_distance):
        check_index = half_distance * distance + index
        support = (bottom_row + 2 * index + 1, bottom_row + 2 * index + 2)
        check_matrix[check_index, support] = 1

    logical_support = np.arange(0, distance**2, distance, dtype=np.int64)
    return check_matrix, logical_support


# Build the color-code check matrix and logical support.
def build_color_code(distance):
    supports = color_code_stabilizer_supports[distance]
    num_qubits = max(max(support) for support in supports) + 1
    check_matrix = np.zeros((len(supports), num_qubits), dtype=np.uint8)
    for check_index, support in enumerate(supports):
        check_matrix[check_index, support] = 1

    logical_support = np.asarray(
        color_code_logical_supports[distance],
        dtype=np.int64,
    )
    return check_matrix, logical_support


# Append the logical observable as the final check-matrix row.
def _augment_with_logical(check_matrix, logical_support):
    augmented = np.zeros(
        (check_matrix.shape[0] + 1, check_matrix.shape[1]),
        dtype=np.uint8,
    )
    augmented[:-1] = check_matrix
    augmented[-1, logical_support] = 1
    return augmented


# Encode each binary matrix row as a 64-bit integer mask.
def _rows_as_bit_masks(binary_matrix):
    if binary_matrix.shape[1] > 64:
        raise ValueError("At most 64 physical qubits are supported.")

    masks = np.zeros(binary_matrix.shape[0], dtype=np.uint64)
    for row_index, row in enumerate(binary_matrix):
        mask = 0
        for column_index in np.flatnonzero(row):
            mask |= 1 << int(column_index)
        masks[row_index] = mask
    return masks


# Compute the Hamming weights of all words in the matrix row space.
def _row_space_weights(binary_matrix):
    row_masks = _rows_as_bit_masks(binary_matrix)
    codewords = np.empty(1 << len(row_masks), dtype=np.uint64)
    codewords[0] = 0

    span_size = 1
    for row_mask in row_masks:
        np.bitwise_xor(
            codewords[:span_size],
            row_mask,
            out=codewords[span_size : 2 * span_size],
        )
        span_size *= 2

    return np.bitwise_count(codewords)


# Apply the normalized inverse Walsh-Hadamard transform in place.
def _inverse_walsh_hadamard(values):
    block_size = 2
    while block_size <= values.size:
        half_size = block_size // 2
        blocks = values.reshape(-1, block_size)
        left = blocks[:, :half_size]
        right = blocks[:, half_size:]
        original_left = left.copy()
        np.add(left, right, out=left)
        np.subtract(original_left, right, out=right)
        block_size *= 2

    values /= values.size
    np.maximum(values, 0.0, out=values)
    values /= values.sum()
    return values


# Precompute the row-space weights needed to evaluate P(S,L).
def prepare_distribution_S_L(check_matrix, logical_support):
    check_and_logical_matrix = _augment_with_logical(
        check_matrix,
        logical_support,
    )
    return _row_space_weights(check_and_logical_matrix)


# Compute the joint probability distribution P(S,L).
def distribution_S_L(row_space_weights_s_l, probability):
    characteristic = np.power(
        1.0 - 2.0 * probability,
        row_space_weights_s_l,
        dtype=np.float64,
    )
    return _inverse_walsh_hadamard(characteristic)


# Marginalize P(S,L) over L to obtain P(S).
def distribution_S(distribution_s_l):
    number_of_syndromes = distribution_s_l.size // 2
    return (
        distribution_s_l[:number_of_syndromes]
        + distribution_s_l[number_of_syndromes:]
    )


# Compute the Shannon entropy of a probability distribution in bits.
def _shannon_entropy(probabilities, chunk_size=1 << 20):
    entropy = 0.0
    for start in range(0, probabilities.size, chunk_size):
        chunk = probabilities[start : start + chunk_size]
        positive = chunk[chunk > 0.0]
        entropy -= np.sum(positive * np.log2(positive))
    return entropy


# Compute the syndrome entropy H(S).
def H_S(distribution_s):
    return _shannon_entropy(distribution_s)


# Compute the joint entropy H(L,S).
def H_L_S(distribution_s_l):
    return _shannon_entropy(distribution_s_l)


# Compute the conditional entropy H(E|L,S).
def H_E_knowing_L_S(check_matrix, probability, distribution_s_l):
    return H_E(check_matrix, probability) - H_L_S(distribution_s_l)


# Evaluate H(S) and H(E|L,S) over all physical error probabilities.
def compute_H_S_and_H_E_knowing_L_S(
    check_matrix,
    logical_support,
    code_name,
    distance,
):
    row_space_weights_s_l = prepare_distribution_S_L(
        check_matrix,
        logical_support,
    )
    h_s_values = np.empty(p_list.size)
    h_e_knowing_l_s_values = np.empty(p_list.size)

    for index, probability in enumerate(p_list):
        print(
            f"{code_name}, d={distance}: "
            f"point {index + 1}/{p_list.size}",
            flush=True,
        )
        distribution_s_l = distribution_S_L(
            row_space_weights_s_l,
            probability,
        )
        distribution_s = distribution_S(distribution_s_l)
        h_s_values[index] = H_S(distribution_s)
        h_e_knowing_l_s_values[index] = H_E_knowing_L_S(
            check_matrix,
            probability,
            distribution_s_l,
        )

    return h_s_values, h_e_knowing_l_s_values


# Compute the independent-check approximation to delta H(S).
def delta_H_S_independent_d5_d7(code_name):
    if code_name == "Surface code":
        return (
            10 * _binary_entropy((1.0 - (1.0 - 2.0 * p_list) ** 4) / 2.0)
            + 2 * _binary_entropy((1.0 - (1.0 - 2.0 * p_list) ** 2) / 2.0)
        )

    return (
        6 * _binary_entropy((1.0 - (1.0 - 2.0 * p_list) ** 6) / 2.0)
        + 3 * _binary_entropy((1.0 - (1.0 - 2.0 * p_list) ** 4) / 2.0)
    )


# Compute all entropy changes from distance 5 to 7 for one code.
def compute_delta_entropies_d5_d7(code_name, code_builder):
    check_d5, logical_d5 = code_builder(5)
    check_d7, logical_d7 = code_builder(7)

    h_s_d5, h_e_knowing_l_s_d5 = compute_H_S_and_H_E_knowing_L_S(
        check_d5,
        logical_d5,
        code_name,
        5,
    )
    h_s_d7, h_e_knowing_l_s_d7 = compute_H_S_and_H_E_knowing_L_S(
        check_d7,
        logical_d7,
        code_name,
        7,
    )

    h_e_d5 = H_E(check_d5, p_list)
    h_e_d7 = H_E(check_d7, p_list)

    return {
        "h_e": h_e_d7 - h_e_d5,
        "h_s": h_s_d7 - h_s_d5,
        "h_s_independent": delta_H_S_independent_d5_d7(code_name),
        "h_e_knowing_l_s": h_e_knowing_l_s_d7 - h_e_knowing_l_s_d5,
    }


# Estimate the intersection of two positive curves on log-log axes.
def _log_log_intersection(x_values, first_curve, second_curve):
    log_difference = np.log(first_curve) - np.log(second_curve)
    crossing_indices = np.flatnonzero(
        log_difference[:-1] * log_difference[1:] <= 0
    )
    if crossing_indices.size == 0:
        return None

    index = crossing_indices[0]
    fraction = -log_difference[index] / (
        log_difference[index + 1] - log_difference[index]
    )
    log_x = np.log(x_values[index]) + fraction * (
        np.log(x_values[index + 1]) - np.log(x_values[index])
    )
    log_y = np.log(first_curve[index]) + fraction * (
        np.log(first_curve[index + 1]) - np.log(first_curve[index])
    )
    return np.exp(log_x), np.exp(log_y)


# Plot both code families' entropy changes and save the PDF.
def plot_delta_entropies_d5_d7(
    surface_code_delta_entropies,
    color_code_delta_entropies,
):
    
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    quantity_styles = {
        "h_e": (r"$\Delta H(E)$", default_colors[1]),
        "h_s": (r"$\Delta H(S)$", default_colors[0]),
        "h_s_independent": (
            r"$\Delta H_{\mathrm{indep.}}(S)$",
            default_colors[0],
        ),
        "h_e_knowing_l_s": (
            r"$\Delta H(E\mid L,S)$",
            default_colors[2],
        ),
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(510 / 3 / 72.27, 510 / 3 / 72.27 * 1.6),
        sharex=True,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.28, top=0.91, hspace=0.38)
    plt.xlim((1e-2,0.5))

    panels = (
        ("Surface code", surface_code_delta_entropies),
        ("Color code", color_code_delta_entropies),
    )

    for ax, (code_name, delta_entropies) in zip(axes, panels):
        for entropy_name, (_, color) in quantity_styles.items():
            ax.plot(
                p_list,
                delta_entropies[entropy_name],
                color=color
            )
        estimated_threshold = _log_log_intersection(
            p_list,
            delta_entropies["h_e"],
            delta_entropies["h_s_independent"],
        )
        if estimated_threshold is not None:
            ax.text(0.17,0.91,"estimated",transform=ax.transAxes,
                ha="left",va="center",fontsize=6,
            )
            ax.text(0.17,0.79,"threshold",transform=ax.transAxes,
                ha="left",va="center",fontsize=6,
            )
            ax.annotate(
                "",
                xy=estimated_threshold,
                xycoords="data",
                xytext=(0.46, 0.84),
                textcoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-|>",
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
                    "mutation_scale": 7,
                    "shrinkA": 0,
                    "shrinkB": 1,
                    "facecolor": "black",
                    "edgecolor": "black"
                },
            )
        ax.axvline(
            threshold, c = 'k',
            linestyle=(0, (3, 2)),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", labelsize=7, pad=2)
        ax.grid(which="major", alpha=0.35)
        ax.set_title(
            code_name,
            fontsize=7,
            pad=4,
        )

    # Reserve a full logarithmic band above the curves for annotations.
    axes[0].set_ylim(top=100)
    axes[1].set_ylim(top=200)
    axes[0].tick_params(axis="x", which="both", labelbottom=False)
    axes[0].text(
        threshold * 1.04,
        0.48,
        r"$p_{\mathrm{th}}$",
        transform=axes[0].get_xaxis_transform(),
        ha="left",
        va="center",
        fontsize=9,
    )
    axes[1].set_xlabel(
        r"$p$",
        fontsize=9,
        labelpad=1.2,
    )
    fig.text(
        0.035,
        0.595,
        r"$\Delta_{5\to7} H$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )

    quantity_handles = [
        line_2d([0], [0], color=color, label=label)
        for label, color in quantity_styles.values()
    ]
    fig.legend(
        handles=quantity_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.015),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.6,
        labelspacing=0.1,
    )

    plt.savefig(
        "entropy_surface_color_code.pdf",
        format="pdf",
        bbox_inches=fig.bbox_inches,
    )
    plt.close(fig)


# Compute the entropy curves and generate the combined PDF figure.
def main():
    surface_code_delta_entropies = compute_delta_entropies_d5_d7(
        "Surface code",
        build_surface_code,
    )
    color_code_delta_entropies = compute_delta_entropies_d5_d7(
        "Color code",
        build_color_code,
    )
    plot_delta_entropies_d5_d7(
        surface_code_delta_entropies,
        color_code_delta_entropies,
    )
    print("Saved figure")


if __name__ == "__main__":
    main()
