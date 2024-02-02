import matplotlib.pyplot as plt
import numpy as np
import json


tests = ['arccos', 'exp','fft', 'inv', 'svd'] # Tests to be shown in graphs
show_test_numbers = True # Show OPS numbers
round_numbers = True # Round to integer numbers

def get_benchmark_data():
    results = {}
    with open('results.json') as f:
        js = json.load(f)
        for bench in js['benchmarks']:
            test_name = bench["name"]
            test_name = test_name[test_name.find('_') + 1:test_name.find('[')]

            key = bench["param"]
            val = bench["stats"]["ops"]

            if round_numbers:
                val = round(val)

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

    width = 0.25
    multiplier = 0
    names = ['numpy', 'dpnp', 'cupy']

    tests = None
    if test_list:
        tests = test_list
    else:
        tests = results.keys()

    tests_values = {}
    x = np.arange(len(tests))

    for name in names:
        tests_values[name] = []

    for test in tests:
        for name in names:
            if name in results[test]:
                tests_values[name].append(results[test][name])
            else:
                tests_values[name].append(np.NaN)

    fig, ax = plt.subplots(layout='constrained')

    for name in names:
        offset = width * multiplier
        rects = ax.bar(x + offset, tests_values[name], width, label=name)
        if show_numbers:
            ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('OPS')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(x + width, tests)
    ax.legend(loc='upper left', ncols=3)

    fig.savefig("img/comparison.png")
    plt.show()
    
def main():
    generate_group_graph(tests, show_test_numbers)

if __name__ == "__main__":
    main()