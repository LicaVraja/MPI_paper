import random
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_sorted_array(n):
    filename = "sorted" + str(n) + ".txt"
    with open(filename, "w") as file:
        for i in range(1, n + 1):
            file.write(f"{i}\n")

def generate_reverse_array(n):
    filename = "reverse_sorted" + str(n) + ".txt"
    with open(filename, "w") as file:
        for i in range(n, 0, -1):
            file.write(f"{i}\n")

def generate_random_array(n):
    filename = "random" + str(n) + ".txt"
    numbers = list(range(1, n + 1))
    random.shuffle(numbers)

    with open(filename, "w") as file:
        for number in numbers:
            file.write(f"{number}\n")


def generate_dataset():
    dataset_dim_array = [100, 1000, 5000, 10000, 25000]
    for dim in dataset_dim_array:
        generate_sorted_array(dim)
        generate_reverse_array(dim)
        generate_random_array(dim)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def iterative_quick_sort(arr):
    stack = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_index = partition(arr, low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))
    return arr


def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def measure_sorting_algorithm(algorithm, data):
    start_time = time.time()
    sorted_data = algorithm(data.copy())
    end_time = time.time()
    return end_time - start_time

def load_data(filename):
    with open(filename, "r") as file:
        data = [int(line.strip()) for line in file]
    return data

def measure_sorting_algorithms(algorithms, datasets):
    results = {algo.__name__: [] for algo in algorithms}

    for data_type, data_files in datasets.items():
        for file in data_files:
            data = load_data(file)
            for algo in algorithms:
                time_taken = measure_sorting_algorithm(algo, data)
                results[algo.__name__].append((data_type, len(data), time_taken))
                print(f"Algorithm: {algo.__name__}, Data type: {data_type}, Size: {len(data)}, Time: {time_taken:.5f}s")
    return results

def save_results(results, filename="sorting_results.txt"):
    with open(filename, "w") as file:
        for algo, data in results.items():
            file.write(f"{algo}:\n")
            for data_type, size, time_taken in data:
                file.write(f"{data_type},{size},{time_taken:.5f}\n")
    print(f"Results saved to {filename}")

def load_results(filename="sorting_results.txt"):
    results = {}
    with open(filename, "r") as file:
        lines = file.readlines()
        current_algo = ""
        for line in lines:
            line = line.strip()
            if line.endswith(":"):
                current_algo = line[:-1]
                results[current_algo] = []
            else:
                data_type, size, time_taken = line.split(',')
                results[current_algo].append((data_type, int(size), float(time_taken)))
    return results

def visualize_results(results):
    for algo, data in results.items():
        plt.figure(figsize=(12, 8))
        for data_type in set(item[0] for item in data):
            filtered_data = [(size, time_taken) for dt, size, time_taken in data if dt == data_type]
            sizes = [item[0] for item in filtered_data]
            times = [item[1] for item in filtered_data]
            plt.plot(sizes, times, marker='o', label=data_type)
        plt.xlabel('Dataset Size')
        plt.ylabel('Execution Time (s)')
        plt.title(f'Performance of {algo}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{algo}_performance.png")
        plt.close()
    print("Plots saved for each algorithm.")

def main():
    generate_dataset()

    datasets = {
        "sorted": [f"sorted{size}.txt" for size in [100, 1000, 5000, 10000, 25000]],
        "reverse_sorted": [f"reverse_sorted{size}.txt" for size in [100, 1000, 5000, 10000, 25000]],
        "random": [f"random{size}.txt" for size in [100, 1000, 5000, 10000, 25000]]
    }

    algorithms = [bubble_sort, quick_sort, iterative_quick_sort, heap_sort, selection_sort, insertion_sort]
    results = measure_sorting_algorithms(algorithms, datasets)
    save_results(results)

    loaded_results = load_results()
    visualize_results(loaded_results)

if __name__ == "__main__":
    main()
