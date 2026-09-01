import random
import json
import argparse
import math
from itertools import product
from collections import Counter

DEFAULT_MAX_COINS = 500
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase = "Now it's {player}'s turn."

# Fixed player names
player1 = "Leo"
player2 = "Sultan"


def save_final_coin_histogram(final_coins, max_remove, output_path, heldout_final_coins=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping final-coin histogram.")
        return

    heldout_final_coins = set(heldout_final_coins or [])
    counts = Counter(final_coins)
    all_xs = set(counts) | heldout_final_coins
    xs = list(range(min(all_xs), max(all_xs) + 1))
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(12, 5))
    plt.bar(xs, ys, width=1.0)
    if heldout_final_coins:
        heldout_xs = sorted(heldout_final_coins)
        plt.scatter(
            heldout_xs,
            [0] * len(heldout_xs),
            marker="|",
            color="red",
            s=80,
            label="held out from train",
        )
        plt.legend()
    plt.xlabel("Coins left before answer move")
    plt.ylabel("Training examples")
    plt.title(f"Training final coin distribution (max_remove={max_remove})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training final-coin histogram to {output_path}")


def save_eval_distribution_plot(initial_coins, final_coins, max_remove, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping eval distribution plot.")
        return

    initial_counts = Counter(initial_coins)
    final_counts = Counter(final_coins)
    initial_xs = list(range(min(initial_counts), max(initial_counts) + 1))
    final_xs = list(range(min(final_counts), max(final_counts) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    axes[0].bar(initial_xs, [initial_counts[x] for x in initial_xs], width=1.0)
    axes[0].set_title(f"Eval initial coin distribution (max_remove={max_remove})")
    axes[0].set_xlabel("Initial coins")
    axes[0].set_ylabel("Eval examples")

    axes[1].bar(final_xs, [final_counts[x] for x in final_xs], width=1.0)
    axes[1].set_title("Eval final coin distribution before answer move")
    axes[1].set_xlabel("Coins left before answer move")
    axes[1].set_ylabel("Eval examples")

    axes[2].scatter(initial_coins, final_coins, s=8, alpha=0.35)
    axes[2].set_title("Eval initial vs. final coin piles")
    axes[2].set_xlabel("Initial coins")
    axes[2].set_ylabel("Coins left before answer move")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved eval initial/final distribution plot to {output_path}")


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return 0


def possible_final_coins(initial_coins, max_remove, min_moves=2, max_moves=4):
    finals = set()
    for n_coins in initial_coins:
        for num_moves in range(min_moves, max_moves + 1):
            for total_removed in range(num_moves, num_moves * max_remove + 1):
                final_coin = n_coins - total_removed
                if final_coin > 0:
                    finals.add(final_coin)
    return finals


def build_moves_by_sum(max_remove, min_moves=2, max_moves=4):
    moves_by_sum = {}
    for num_moves in range(min_moves, max_moves + 1):
        for moves in product(range(1, max_remove + 1), repeat=num_moves):
            moves_by_sum.setdefault(sum(moves), []).append(moves)
    return moves_by_sum


def build_moves_by_count_and_sum(max_remove, min_moves=2, max_moves=4):
    moves_by_count_and_sum = {
        num_moves: {}
        for num_moves in range(min_moves, max_moves + 1)
    }
    for num_moves in range(min_moves, max_moves + 1):
        for moves in product(range(1, max_remove + 1), repeat=num_moves):
            moves_by_count_and_sum[num_moves].setdefault(sum(moves), []).append(moves)
    return moves_by_count_and_sum


def select_stratified_final_holdout(
    train_possible_final,
    eval_possible_final,
    eval_initial_coins,
    moves_by_sum,
    max_remove,
    holdout_frac,
):
    mod = max_remove + 1
    total_holdout = max(1, math.ceil(len(train_possible_final) * holdout_frac))
    eligible_final = train_possible_final & eval_possible_final

    raw_quotas = {
        residue: sum(1 for final_coin in train_possible_final if final_coin % mod == residue) * holdout_frac
        for residue in range(mod)
    }
    quotas = {residue: math.floor(raw_quotas[residue]) for residue in range(mod)}

    residues = list(range(mod))
    random.shuffle(residues)
    residues.sort(key=lambda residue: raw_quotas[residue] - quotas[residue], reverse=True)
    for residue in residues[: total_holdout - sum(quotas.values())]:
        quotas[residue] += 1

    finals_for_initial = {
        n_coins: [
            n_coins - removed
            for removed in moves_by_sum
            if n_coins - removed in eligible_final
        ]
        for n_coins in eval_initial_coins
    }
    uncovered = [n_coins for n_coins, finals in finals_for_initial.items() if not finals]
    if uncovered:
        raise ValueError(f"Some eval initial piles cannot reach any held-out final candidate: {uncovered}")

    initial_items = list(eval_initial_coins)
    final_items = list(eligible_final)
    random.shuffle(initial_items)
    random.shuffle(final_items)
    source = 0
    initial_offset = 1
    final_offset = initial_offset + len(initial_items)
    residue_offset = final_offset + len(final_items)
    sink = residue_offset + mod
    flow_graph = Dinic(sink + 1)

    for idx, _ in enumerate(initial_items):
        flow_graph.add_edge(source, initial_offset + idx, 1)

    final_index = {final_coin: idx for idx, final_coin in enumerate(final_items)}
    for initial_idx, n_coins in enumerate(initial_items):
        finals = list(finals_for_initial[n_coins])
        random.shuffle(finals)
        for final_coin in finals:
            flow_graph.add_edge(
                initial_offset + initial_idx,
                final_offset + final_index[final_coin],
                1,
            )

    final_edges = {}
    for final_coin, idx in final_index.items():
        edge = flow_graph.add_edge(
            final_offset + idx,
            residue_offset + (final_coin % mod),
            1,
        )
        final_edges[final_coin] = edge

    for residue, quota in quotas.items():
        flow_graph.add_edge(residue_offset + residue, sink, quota)

    matched = flow_graph.max_flow(source, sink)
    if matched != len(initial_items):
        raise RuntimeError(
            f"Could only match {matched}/{len(initial_items)} eval initials to unique held-out final piles."
        )

    holdout = {
        final_coin
        for final_coin, edge in final_edges.items()
        if edge[2][1] > 0
    }
    remaining_quotas = dict(quotas)
    for final_coin in holdout:
        remaining_quotas[final_coin % mod] -= 1

    for residue, quota in remaining_quotas.items():
        eligible = [
            final_coin
            for final_coin in eligible_final
            if final_coin not in holdout and final_coin % mod == residue
        ]
        if len(eligible) < quota:
            raise ValueError(
                f"Need {quota} additional held-out final piles with residue {residue}, "
                f"but only {len(eligible)} are available."
            )
        random.shuffle(eligible)
        holdout.update(eligible[:quota])

    return holdout


def balanced_final_values(final_coins, max_remove, n_examples):
    mod = max_remove + 1
    buckets = {
        residue: [final_coin for final_coin in final_coins if final_coin % mod == residue]
        for residue in range(mod)
    }
    missing = [residue for residue, values in buckets.items() if not values]
    if missing:
        raise ValueError(f"No final-pile candidates for residues: {missing}")

    residues = list(range(mod))
    random.shuffle(residues)
    counts = {residue: n_examples // mod for residue in range(mod)}
    for residue in residues[: n_examples % mod]:
        counts[residue] += 1

    final_values = []
    for residue, count in counts.items():
        remaining = count
        bucket = buckets[residue]
        while remaining > 0:
            cycle = bucket[:]
            random.shuffle(cycle)
            take = min(remaining, len(cycle))
            final_values.extend(cycle[:take])
            remaining -= take

    random.shuffle(final_values)
    return final_values


def generate_nim_example_from_final(max_remove, final_coin, initial_coins, moves_by_sum):
    valid_removed = [
        removed
        for removed in moves_by_sum
        if final_coin + removed in initial_coins
    ]
    if not valid_removed:
        raise ValueError(f"Final pile {final_coin} is not reachable from the requested initial split.")

    removed = random.choice(valid_removed)
    moves = random.choice(moves_by_sum[removed])
    n_coins = final_coin + removed

    # ensure enough coins so game doesn't end immediately
    current = n_coins
    trace = []
    turn = 0  # 0 is player1, 1 is player2

    for amt in moves:
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn
    if current != final_coin:
        raise ValueError(f"Trace construction failed: expected {final_coin}, got {current}")

    move = best_move(current, max_remove)
    players = [player1, player2]

    # build trace text
    trace_lines = []
    for idx, amt in trace:
        actor = players[idx]
        plural = "s" if amt > 1 else ""
        trace_lines.append(f"{actor} {take_verb} {amt} {coin_name}{plural}.")

    # build prompt
    desc = f"You are playing the game of {game_name}. There are {n_coins} {coin_name}s.\n"
    desc += f"{player1} and {player2} take turns.\n"
    desc += f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.\n\n"

    if trace_lines:
        desc += "So far:\n" + "\n".join(trace_lines) + "\n"

    desc += turn_phrase.format(player=players[turn]) + "\n\n"

    answer = f"{move}"
    return ({"prompt": desc.strip(), "answer": answer}, current)


def generate_nim_example_from_components(max_remove, n_coins, final_coin, moves):
    current = n_coins
    trace = []
    turn = 0  # 0 is player1, 1 is player2

    for amt in moves:
        trace.append((turn, amt))
        current -= amt
        turn = 1 - turn
    if current != final_coin:
        raise ValueError(f"Trace construction failed: expected {final_coin}, got {current}")

    move = best_move(current, max_remove)
    players = [player1, player2]

    trace_lines = []
    for idx, amt in trace:
        actor = players[idx]
        plural = "s" if amt > 1 else ""
        trace_lines.append(f"{actor} {take_verb} {amt} {coin_name}{plural}.")

    desc = f"You are playing the game of {game_name}. There are {n_coins} {coin_name}s.\n"
    desc += f"{player1} and {player2} take turns.\n"
    desc += f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.\n\n"
    desc += "So far:\n" + "\n".join(trace_lines) + "\n"
    desc += turn_phrase.format(player=players[turn]) + "\n\n"

    return {"prompt": desc.strip(), "answer": f"{move}"}


def balanced_targets(items, total):
    items = list(items)
    random.shuffle(items)
    base_count = total // len(items)
    counts = {item: base_count for item in items}
    for item in items[: total % len(items)]:
        counts[item] += 1
    return counts


class Dinic:
    def __init__(self, size):
        self.graph = [[] for _ in range(size)]

    def add_edge(self, src, dst, capacity):
        forward = [dst, capacity, None]
        backward = [src, 0, forward]
        forward[2] = backward
        self.graph[src].append(forward)
        self.graph[dst].append(backward)
        return forward

    def max_flow(self, source, sink):
        flow = 0
        while True:
            level = [-1] * len(self.graph)
            queue = [source]
            level[source] = 0
            for node in queue:
                for dst, capacity, _ in self.graph[node]:
                    if capacity > 0 and level[dst] < 0:
                        level[dst] = level[node] + 1
                        queue.append(dst)
            if level[sink] < 0:
                return flow

            progress = [0] * len(self.graph)

            def send(node, amount):
                if node == sink:
                    return amount
                while progress[node] < len(self.graph[node]):
                    edge = self.graph[node][progress[node]]
                    dst, capacity, reverse = edge
                    if capacity > 0 and level[node] + 1 == level[dst]:
                        pushed = send(dst, min(amount, capacity))
                        if pushed:
                            edge[1] -= pushed
                            reverse[1] += pushed
                            return pushed
                    progress[node] += 1
                return 0

            while True:
                pushed = send(source, 10**9)
                if not pushed:
                    break
                flow += pushed


def assign_initial_final_counts(initial_targets, final_coins, reachable_finals, n_examples):
    initial_items = list(initial_targets)
    final_items = list(final_coins)
    random.shuffle(initial_items)
    random.shuffle(final_items)

    source = 0
    initial_offset = 1
    final_offset = initial_offset + len(initial_items)
    sink = final_offset + len(final_items)
    super_source = sink + 1
    super_sink = sink + 2
    graph_size = super_sink + 1
    final_mean = n_examples / len(final_items)

    for slack in range(0, 20):
        final_low = max(0, math.floor(final_mean) - slack)
        final_high = math.ceil(final_mean) + slack
        if final_low * len(final_items) > n_examples or final_high * len(final_items) < n_examples:
            continue

        flow_graph = Dinic(graph_size)
        demands = [0] * graph_size
        edge_lookup = {}

        def add_lower_bound_edge(src, dst, low, high):
            edge = flow_graph.add_edge(src, dst, high - low)
            demands[src] -= low
            demands[dst] += low
            return edge, low

        for idx, n_coins in enumerate(initial_items):
            add_lower_bound_edge(source, initial_offset + idx, initial_targets[n_coins], initial_targets[n_coins])

        final_index = {final_coin: idx for idx, final_coin in enumerate(final_items)}
        for initial_idx, n_coins in enumerate(initial_items):
            finals = list(reachable_finals[n_coins])
            random.shuffle(finals)
            for final_coin in finals:
                edge, low = add_lower_bound_edge(
                    initial_offset + initial_idx,
                    final_offset + final_index[final_coin],
                    0,
                    n_examples,
                )
                edge_lookup[(n_coins, final_coin)] = (edge, low)

        for idx, final_coin in enumerate(final_items):
            add_lower_bound_edge(final_offset + idx, sink, final_low, final_high)

        add_lower_bound_edge(sink, source, 0, n_examples)

        required = 0
        for node, demand in enumerate(demands):
            if demand > 0:
                flow_graph.add_edge(super_source, node, demand)
                required += demand
            elif demand < 0:
                flow_graph.add_edge(node, super_sink, -demand)

        actual = flow_graph.max_flow(super_source, super_sink)
        if actual != required:
            continue

        assignments = []
        for (n_coins, final_coin), (edge, low) in edge_lookup.items():
            assigned = low + edge[2][1]
            if assigned:
                assignments.append((n_coins, final_coin, assigned))
        return assignments, (final_low, final_high)

    raise RuntimeError(
        "Could not balance initial piles and exact final piles within the tested final-count tolerance."
    )


def assign_move_counts(pair_assignments, moves_by_count_and_sum, n_examples):
    move_targets = balanced_targets([2, 3, 4], n_examples)
    source = 0
    pair_offset = 1
    move_offset = pair_offset + len(pair_assignments)
    sink = move_offset + 3
    flow_graph = Dinic(sink + 1)
    edge_lookup = {}

    for idx, (n_coins, final_coin, count) in enumerate(pair_assignments):
        flow_graph.add_edge(source, pair_offset + idx, count)
        removed = n_coins - final_coin
        for move_count in [2, 3, 4]:
            if removed in moves_by_count_and_sum[move_count]:
                edge = flow_graph.add_edge(
                    pair_offset + idx,
                    move_offset + (move_count - 2),
                    count,
                )
                edge_lookup[(idx, move_count)] = edge

    for move_count in [2, 3, 4]:
        flow_graph.add_edge(
            move_offset + (move_count - 2),
            sink,
            move_targets[move_count],
        )

    actual = flow_graph.max_flow(source, sink)
    if actual != n_examples:
        raise RuntimeError(
            f"Could only assign move counts for {actual}/{n_examples} examples."
        )

    assignments = []
    for (idx, move_count), edge in edge_lookup.items():
        assigned = edge[2][1]
        if assigned:
            n_coins, final_coin, _ = pair_assignments[idx]
            assignments.append((n_coins, final_coin, move_count, assigned))
    return assignments, move_targets


def generate_balanced_examples(
    max_remove,
    initial_coins,
    final_coins,
    moves_by_sum,
    moves_by_count_and_sum,
    n_examples,
):
    initial_coins = list(initial_coins)
    final_coins = set(final_coins)

    candidates = {}
    reachable_finals = {}
    for n_coins in initial_coins:
        for removed, move_options in moves_by_sum.items():
            final_coin = n_coins - removed
            if final_coin not in final_coins:
                continue
            candidates.setdefault((n_coins, final_coin), []).append(removed)
            reachable_finals.setdefault(n_coins, set()).add(final_coin)

    missing_initials = [n_coins for n_coins in initial_coins if n_coins not in reachable_finals]
    if missing_initials:
        raise ValueError(f"Some initial piles cannot reach any allowed final pile: {missing_initials}")

    initial_targets = balanced_targets(initial_coins, n_examples)
    assignments, final_bounds = assign_initial_final_counts(
        initial_targets,
        final_coins,
        reachable_finals,
        n_examples,
    )
    move_assignments, move_targets = assign_move_counts(
        assignments,
        moves_by_count_and_sum,
        n_examples,
    )
    examples = []
    final_values = []
    initial_values = []

    expanded = []
    for n_coins, final_coin, move_count, count in move_assignments:
        expanded.extend([(n_coins, final_coin, move_count)] * count)
    random.shuffle(expanded)

    for n_coins, final_coin, move_count in expanded:
        removed = n_coins - final_coin
        moves = random.choice(moves_by_count_and_sum[move_count][removed])

        examples.append(generate_nim_example_from_components(max_remove, n_coins, final_coin, moves))
        final_values.append(final_coin)
        initial_values.append(n_coins)

    return examples, initial_values, final_values, final_bounds, move_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-remove", type=int, required=True,
                        help="Maximum number of coins that can be taken in one move (defines modulus m = max_remove+1).")
    parser.add_argument("--n-train", type=int, default=15000,
                        help="Number of training examples to generate.")
    parser.add_argument("--n-eval", type=int, default=2000,
                        help="Number of eval examples to generate.")
    parser.add_argument("--max-coins", type=int, default=DEFAULT_MAX_COINS,
                        help="Maximum initial coin pile to sample from.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--final-holdout-frac", type=float, default=0.15,
                        help="Fraction of train-reachable final coin piles to hold out from train.")
    args = parser.parse_args()

    random.seed(args.seed)
    m = args.max_remove

    # ---- train set ----
    min_coins = (8 + 1) * (4 + 1)
    if args.max_coins < min_coins:
        raise ValueError(f"--max-coins must be >= {min_coins}, got {args.max_coins}")
    nums = list(range(min_coins, args.max_coins + 1))
    random.shuffle(nums)

    split_idx = int(len(nums) * 0.85)

    train_initial = nums[:split_idx]
    eval_initial = nums[split_idx:]
    train_possible_final = possible_final_coins(train_initial, m)
    eval_possible_final = possible_final_coins(eval_initial, m)
    if not train_possible_final:
        raise ValueError("No train-reachable final coin piles are available to hold out.")

    moves_by_sum = build_moves_by_sum(m)
    moves_by_count_and_sum = build_moves_by_count_and_sum(m)
    train_final_holdout = select_stratified_final_holdout(
        train_possible_final,
        eval_possible_final,
        eval_initial,
        moves_by_sum,
        m,
        args.final_holdout_frac,
    )
    train_final_pool = train_possible_final - train_final_holdout

    train_dataset, train_initial_values, train_final_values, train_final_bounds, train_move_targets = generate_balanced_examples(
        m,
        train_initial,
        train_final_pool,
        moves_by_sum,
        moves_by_count_and_sum,
        args.n_train,
    )
    train_final = set(train_final_values)

    random.shuffle(train_dataset)

    train_filename = f"{m}_train.jsonl"
    with open(train_filename, "w") as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"length of train_final: {len(train_final)}")
    print(f"min of train_final: {min(train_final)}")
    print(f"max of train_final: {max(train_final)}")
    print(f"length of train_final_holdout: {len(train_final_holdout)}")
    print(f"train final count bounds: {train_final_bounds[0]}..{train_final_bounds[1]}")
    print(f"train move count targets: {train_move_targets}")
    histogram_filename = f"new_{m}_train_final_coin_hist.png"
    save_final_coin_histogram(train_final_values, m, histogram_filename, train_final_holdout)
    
    print(f"length of eval_initial: {len(eval_initial)}")

    # ---- eval set (no initial or final pile overlap) ----
    eval_dataset, eval_initial_values, eval_final_values, eval_final_bounds, eval_move_targets = generate_balanced_examples(
        m,
        eval_initial,
        train_final_holdout,
        moves_by_sum,
        moves_by_count_and_sum,
        args.n_eval,
    )
    random.shuffle(eval_dataset)

    eval_filename = f"{m}_eval.jsonl"
    with open(eval_filename, "w") as f:
        for item in eval_dataset:
            f.write(json.dumps(item) + "\n")
    eval_distribution_filename = f"new_{m}_eval_initial_final_dist.png"
    print(f"eval final count bounds: {eval_final_bounds[0]}..{eval_final_bounds[1]}")
    print(f"eval move count targets: {eval_move_targets}")
    save_eval_distribution_plot(eval_initial_values, eval_final_values, m, eval_distribution_filename)

    print(f"Generated {train_filename} (n_train={args.n_train}), {eval_filename} (n_eval={args.n_eval})")


if __name__ == "__main__":
    main()
