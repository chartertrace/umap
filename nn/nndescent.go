// Package nn provides nearest neighbor search algorithms for UMAP.
// This includes the NNDescent algorithm for approximate k-NN graph construction.
package nn

import (
	"github.com/nozzle/umap/distance"
	"github.com/nozzle/umap/internal/heap"
	"github.com/nozzle/umap/internal/parallel"
	"github.com/nozzle/umap/internal/rand"
)

// KNNGraph represents a k-nearest neighbor graph.
type KNNGraph struct {
	Indices   [][]int32   // [n_samples][k] neighbor indices
	Distances [][]float32 // [n_samples][k] neighbor distances
	N         int         // number of samples
	K         int         // number of neighbors per sample
}

// NNDescentConfig configures the NNDescent algorithm.
type NNDescentConfig struct {
	// K is the number of neighbors to find
	K int

	// MaxIterations is the maximum number of NNDescent iterations
	MaxIterations int

	// Delta is the early termination threshold (fraction of updated edges)
	Delta float32

	// Rho is the sampling rate for candidate pairs
	Rho float32

	// Metric is the distance metric to use
	Metric string

	// Seed for random number generation
	Seed int64

	// NumWorkers for parallel processing (0 = auto)
	NumWorkers int

	// Verbose enables progress output
	Verbose bool
}

// DefaultNNDescentConfig returns default configuration.
func DefaultNNDescentConfig() NNDescentConfig {
	return NNDescentConfig{
		K:             15,
		MaxIterations: 10,
		Delta:         0.001,
		Rho:           0.5,
		Metric:        "euclidean",
		Seed:          42,
		NumWorkers:    0,
		Verbose:       false,
	}
}

// NNDescent builds an approximate k-NN graph using the NNDescent algorithm.
// This is an iterative algorithm that refines random initial neighbors by
// exploring "neighbors of neighbors".
func NNDescent(data [][]float32, config NNDescentConfig) *KNNGraph {
	n := len(data)
	k := config.K
	if k >= n {
		k = n - 1
	}

	distFunc, ok := distance.Get(config.Metric)
	if !ok {
		distFunc = distance.Euclidean
	}

	numWorkers := config.NumWorkers
	if numWorkers <= 0 {
		numWorkers = parallel.NumWorkers()
	}

	// Initialize random state
	rng := rand.New(config.Seed)

	// Initialize with random neighbors
	indices := make([][]int32, n)
	distances := make([][]float32, n)
	flags := make([][]uint8, n)

	for i := range indices {
		indices[i] = make([]int32, k)
		distances[i] = make([]float32, k)
		flags[i] = make([]uint8, k)

		// Initialize with sentinel values
		for j := range indices[i] {
			indices[i][j] = -1
			distances[i][j] = 1e30
			flags[i][j] = 1 // mark as new
		}
	}

	// Random initialization: pick k random neighbors for each point
	initializeRandomNeighbors(data, indices, distances, flags, distFunc, &rng, numWorkers)

	// Build reverse neighbor lists for efficient updates
	oldCandidates := make([][]int32, n)
	newCandidates := make([][]int32, n)
	for i := range oldCandidates {
		oldCandidates[i] = make([]int32, 0, k*2)
		newCandidates[i] = make([]int32, 0, k*2)
	}

	// NNDescent iterations
	for iter := 0; iter < config.MaxIterations; iter++ {
		// Build candidate lists (split into old and new)
		for i := range oldCandidates {
			oldCandidates[i] = oldCandidates[i][:0]
			newCandidates[i] = newCandidates[i][:0]
		}

		for i := range n {
			for j := 0; j < k; j++ {
				neighbor := indices[i][j]
				if neighbor < 0 {
					continue
				}
				if flags[i][j] == 1 { // new
					newCandidates[i] = append(newCandidates[i], neighbor)
					// Also add to reverse direction
					if len(newCandidates[neighbor]) < k*2 {
						newCandidates[neighbor] = append(newCandidates[neighbor], int32(i))
					}
				} else { // old
					oldCandidates[i] = append(oldCandidates[i], neighbor)
				}
			}
		}

		// Sample candidates based on rho
		for i := range newCandidates {
			newCandidates[i] = sampleCandidates(newCandidates[i], config.Rho, &rng)
			oldCandidates[i] = sampleCandidates(oldCandidates[i], config.Rho, &rng)
		}

		// Mark all current neighbors as old
		for i := range n {
			for j := 0; j < k; j++ {
				flags[i][j] = 0
			}
		}

		// Update neighbors by exploring candidates
		updates := nnDescentUpdate(data, indices, distances, flags,
			oldCandidates, newCandidates, distFunc, numWorkers)

		// Check convergence
		updateRate := float32(updates) / float32(n*k)
		if updateRate < config.Delta {
			break
		}
	}

	// Sort results by distance
	for i := range n {
		heap.DeheapSort(indices[i], distances[i], k)
	}

	return &KNNGraph{
		Indices:   indices,
		Distances: distances,
		N:         n,
		K:         k,
	}
}

// initializeRandomNeighbors initializes each point with k random neighbors.
func initializeRandomNeighbors(
	data [][]float32,
	indices [][]int32,
	distances [][]float32,
	flags [][]uint8,
	distFunc distance.Func,
	rng *rand.State,
	numWorkers int,
) {
	n := len(data)
	k := len(indices[0])

	// Generate random neighbors for each point
	parallel.ParallelFor(0, n, numWorkers, func(i int) {
		// Create a local RNG for this point to avoid contention
		localRng := rand.New(int64(i) + rng[0])

		// Generate k unique random neighbors
		selected := make(map[int32]bool, k)
		for len(selected) < k && len(selected) < n-1 {
			j := rand.Intn(&localRng, n)
			if j == i {
				continue
			}
			if selected[int32(j)] {
				continue
			}
			selected[int32(j)] = true

			dist := distFunc(data[i], data[j])
			idx := len(selected) - 1
			indices[i][idx] = int32(j)
			distances[i][idx] = dist
			flags[i][idx] = 1 // new
		}

		// Build heap from initial neighbors
		h := heap.NewFromSlices(indices[i], distances[i], flags[i])
		_ = h // heap is already built
	})
}

// nnDescentUpdate performs one round of NNDescent updates.
// Returns the number of updates made.
func nnDescentUpdate(
	data [][]float32,
	indices [][]int32,
	distances [][]float32,
	flags [][]uint8,
	oldCandidates [][]int32,
	newCandidates [][]int32,
	distFunc distance.Func,
	numWorkers int,
) int {
	n := len(data)
	k := len(indices[0])

	// Count updates per worker
	updateCounts := make([]int, numWorkers)

	parallel.ParallelFor(0, n, numWorkers, func(i int) {
		workerID := i * numWorkers / n

		// Get candidates for point i
		newCands := newCandidates[i]
		oldCands := oldCandidates[i]

		// Compare all new-new pairs and new-old pairs
		for _, p1 := range newCands {
			// Compare with other new candidates
			for _, p2 := range newCands {
				if p1 >= p2 {
					continue // avoid duplicates
				}
				if tryUpdateNeighbors(data, indices, distances, flags, k, int(p1), int(p2), distFunc) {
					updateCounts[workerID]++
				}
			}

			// Compare with old candidates
			for _, p2 := range oldCands {
				if p1 == p2 {
					continue
				}
				if tryUpdateNeighbors(data, indices, distances, flags, k, int(p1), int(p2), distFunc) {
					updateCounts[workerID]++
				}
			}
		}
	})

	// Sum updates
	total := 0
	for _, c := range updateCounts {
		total += c
	}
	return total
}

// tryUpdateNeighbors tries to add point j as a neighbor of point i (and vice versa).
// Returns true if any update was made.
func tryUpdateNeighbors(
	data [][]float32,
	indices [][]int32,
	distances [][]float32,
	flags [][]uint8,
	k int,
	i, j int,
	distFunc distance.Func,
) bool {
	dist := distFunc(data[i], data[j])

	updated := heap.FlaggedHeapPush(indices[i], distances[i], flags[i], k, int32(j), dist, 1)

	// Try to add i as neighbor of j
	if heap.FlaggedHeapPush(indices[j], distances[j], flags[j], k, int32(i), dist, 1) {
		updated = true
	}

	return updated
}

// sampleCandidates randomly samples a subset of candidates based on rho.
func sampleCandidates(candidates []int32, rho float32, rng *rand.State) []int32 {
	if rho >= 1.0 || len(candidates) == 0 {
		return candidates
	}

	targetSize := max(int(float32(len(candidates))*rho), 1)
	if targetSize >= len(candidates) {
		return candidates
	}

	// Shuffle and take first targetSize elements
	rand.Shuffle(rng, candidates)
	return candidates[:targetSize]
}

// NearestK returns the indices and distances of the k points in data closest
// to query under distFunc, sorted nearest-first. It is a single-query brute
// force search intended for projecting out-of-sample points; for whole-dataset
// graph construction use BruteForceKNN or NNDescent instead.
func NearestK(data [][]float32, query []float32, k int, distFunc distance.Func) ([]int32, []float32) {
	n := len(data)
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil, nil
	}

	idx := make([]int32, k)
	dist := make([]float32, k)
	for i := range idx {
		idx[i] = -1
		dist[i] = 1e30
	}

	for j := range data {
		heap.SimpleHeapPush(idx, dist, k, int32(j), distFunc(query, data[j]))
	}
	heap.DeheapSort(idx, dist, k)
	return idx, dist
}

// BruteForceKNN computes exact k-NN using brute force.
// This is used for small datasets or as a fallback.
// Note: k includes self as the first neighbor (distance 0) to match Python sklearn behavior.
func BruteForceKNN(data [][]float32, k int, metric string) *KNNGraph {
	n := len(data)
	if k > n {
		k = n
	}

	distFunc, ok := distance.Get(metric)
	if !ok {
		distFunc = distance.Euclidean
	}

	numWorkers := parallel.NumWorkers()

	indices := make([][]int32, n)
	distances := make([][]float32, n)

	parallel.ParallelFor(0, n, numWorkers, func(i int) {
		indices[i] = make([]int32, k)
		distances[i] = make([]float32, k)

		// Initialize heap with sentinel values
		for j := 0; j < k; j++ {
			indices[i][j] = -1
			distances[i][j] = 1e30
		}

		// Include self as first neighbor (distance 0) - matches Python sklearn
		heap.SimpleHeapPush(indices[i], distances[i], k, int32(i), 0)

		// Compute distance to all other points
		for j := range n {
			if i == j {
				continue
			}
			dist := distFunc(data[i], data[j])
			heap.SimpleHeapPush(indices[i], distances[i], k, int32(j), dist)
		}

		// Sort by distance
		heap.DeheapSort(indices[i], distances[i], k)
	})

	return &KNNGraph{
		Indices:   indices,
		Distances: distances,
		N:         n,
		K:         k,
	}
}
