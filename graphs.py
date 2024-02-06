import matplotlib.pyplot as plt
import numpy as np
import json

PKG_NAMES = ['numpy', 'arrayfire', 'cupy', 'dpnp'] # package list in graph order
tests = ['group_elementwise', 'pi', 'black_scholes', 'fft', 'inv', 'svd'] # Tests to be shown in graphs
show_test_numbers = True # Show Speedup numbers
round_numbers = 1 # Round to digits after decimal

def get_benchmark_data():
    results = {}
    with open('results.json') as f:
        js = json.load(f)
        for bench in js['benchmarks']:
            test_name = bench["name"]
            test_name = test_name[test_name.find('_') + 1:test_name.find('[')]

            key = bench["param"]
            val = bench["stats"]["ops"]

            if test_name not in results:
                results[test_name] = { key : val }
            else:
                results[test_name][key] = val

    return results

def create_graph(test_name, test_results):
    names = []
    values = []
    for name in test_results:
        names.append(name)
        values.append(test_results[name])

    bar = plt.bar(names, values)
    plt.title(test_name)

    plt.savefig("img/" + test_name + ".png")
    plt.close()

def generate_individual_graphs():
    results = get_benchmark_data()

    for test in results:
        create_graph(test, results[test])


def generate_group_graph(test_list = None, show_numbers = False):
    results = get_benchmark_data()

    width = 1 / (1 + len(PKG_NAMES))
    multiplier = 0

    tests = None
    if test_list:
        tests = test_list
    else:
        tests = results.keys()

    tests_values = {}
    x = np.arange(len(tests))

    for name in PKG_NAMES:
        tests_values[name] = []

    max_val = 1
    for test in tests:
        for name in PKG_NAMES:
            base_value = results[test]["numpy"]
            if name in results[test]:
                val = results[test][name] / base_value

                if round_numbers:
                    val = round(val, round_numbers)

                if max_val < val:
                    max_val = val

                tests_values[name].append(val)
            else:
                tests_values[name].append(np.NaN)

    fig, ax = plt.subplots(layout='constrained')

    for name in PKG_NAMES:
        offset = width * multiplier
        rects = ax.bar(x + offset, tests_values[name], width, label=name)
        if show_numbers:
            ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Speedup')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(x + width, tests)
    ax.set_ylim([0.0, max_val * 1.25])
    ax.legend(loc='upper left', ncols=len(PKG_NAMES))

    fig.savefig("img/comparison.png")
    plt.show()
    
def main():
    generate_group_graph(tests, show_test_numbers)

if __name__ == "__main__":
    main()