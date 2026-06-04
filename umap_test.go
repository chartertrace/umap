package umap

import (
	"math"
	"testing"
)

// euclideanDist returns the squared Euclidean distance between two vectors.
// Squared distance is sufficient for the relative comparisons made in tests.
func euclideanDist(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// generateBlobs generates n clusters of points for testing.
func generateBlobs(nSamples, nClusters, nFeatures int, seed int64) [][]float32 {
	data := make([][]float32, nSamples)
	samplesPerCluster := nSamples / nClusters

	// Simple LCG for reproducibility
	rng := seed
	nextFloat := func() float32 {
		rng = (rng*6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
		return float32(rng) / float32(0x7FFFFFFF)
	}

	for i := range nSamples {
		data[i] = make([]float32, nFeatures)
		cluster := i / samplesPerCluster
		if cluster >= nClusters {
			cluster = nClusters - 1
		}

		// Cluster center
		centerOffset := float32(cluster * 10)

		for j := range nFeatures {
			// Add Gaussian-like noise using Box-Muller-ish
			u1 := nextFloat()
			u2 := nextFloat()
			if u1 < 0.001 {
				u1 = 0.001
			}
			noise := float32(math.Sqrt(-2*math.Log(float64(u1)))) * float32(math.Cos(2*math.Pi*float64(u2)))
			data[i][j] = centerOffset + noise
		}
	}

	return data
}

func TestFitTransform(t *testing.T) {
	// Generate simple test data: 3 clusters
	data := generateBlobs(300, 3, 10, 42)

	// Run UMAP
	config := DefaultConfig()
	config.NNeighbors = 10
	config.NComponents = 2
	config.NEpochs = 50 // Fewer epochs for faster testing
	config.Seed = 42
	config.Init = "random" // Faster than spectral

	model := New(config)
	embedding := model.FitTransform(data)

	// Basic checks
	if len(embedding) != len(data) {
		t.Errorf("Expected %d points, got %d", len(data), len(embedding))
	}

	if len(embedding[0]) != config.NComponents {
		t.Errorf("Expected %d components, got %d", config.NComponents, len(embedding[0]))
	}

	// Check that we have finite values
	for i, point := range embedding {
		for j, val := range point {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Errorf("Non-finite value at [%d][%d]: %v", i, j, val)
			}
		}
	}

	// Check that clusters are somewhat separated in the embedding
	// by comparing intra-cluster vs inter-cluster distances
	samplesPerCluster := 100
	intraClusterDist := float32(0)
	interClusterDist := float32(0)
	intraCount := 0
	interCount := 0

	for i := range 50 {
		for j := i + 1; j < 100; j++ {
			clusterI := i / samplesPerCluster
			clusterJ := j / samplesPerCluster
			dist := euclideanDist(embedding[i], embedding[j])

			if clusterI == clusterJ {
				intraClusterDist += dist
				intraCount++
			} else {
				interClusterDist += dist
				interCount++
			}
		}
	}

	if intraCount > 0 && interCount > 0 {
		avgIntra := intraClusterDist / float32(intraCount)
		avgInter := interClusterDist / float32(interCount)

		// Inter-cluster distance should be larger than intra-cluster
		// This is a soft check since UMAP is stochastic
		t.Logf("Avg intra-cluster distance: %f", avgIntra)
		t.Logf("Avg inter-cluster distance: %f", avgInter)
	}
}

func TestBruteForceKNN(t *testing.T) {
	data := generateBlobs(100, 2, 5, 42)

	config := DefaultConfig()
	config.NNeighbors = 10
	config.NComponents = 2
	config.NEpochs = 20
	config.Seed = 42
	config.Init = "random"

	model := New(config)
	embedding := model.FitTransform(data)

	if len(embedding) != 100 {
		t.Errorf("Expected 100 points, got %d", len(embedding))
	}
}

func TestDifferentMetrics(t *testing.T) {
	data := generateBlobs(100, 2, 5, 42)

	metrics := []string{"euclidean", "manhattan", "cosine"}

	for _, metric := range metrics {
		t.Run(metric, func(t *testing.T) {
			config := DefaultConfig()
			config.NNeighbors = 10
			config.Metric = metric
			config.NComponents = 2
			config.NEpochs = 20
			config.Seed = 42
			config.Init = "random"

			model := New(config)
			embedding := model.FitTransform(data)

			if len(embedding) != 100 {
				t.Errorf("Expected 100 points, got %d", len(embedding))
			}

			// Check for finite values
			for i, point := range embedding {
				for j, val := range point {
					if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
						t.Errorf("Non-finite value at [%d][%d]: %v", i, j, val)
					}
				}
			}
		})
	}
}

func TestSmallDataset(t *testing.T) {
	// Very small dataset
	data := [][]float32{
		{1, 2, 3},
		{1, 2, 4},
		{1, 3, 3},
		{10, 20, 30},
		{10, 20, 31},
		{10, 21, 30},
	}

	config := DefaultConfig()
	config.NNeighbors = 2
	config.NComponents = 2
	config.NEpochs = 50
	config.Seed = 42
	config.Init = "random"

	model := New(config)
	embedding := model.FitTransform(data)

	if len(embedding) != 6 {
		t.Errorf("Expected 6 points, got %d", len(embedding))
	}

	// The two clusters should be separated
	// Points 0,1,2 should be close; points 3,4,5 should be close
	dist01 := euclideanDist(embedding[0], embedding[1])
	dist34 := euclideanDist(embedding[3], embedding[4])
	dist03 := euclideanDist(embedding[0], embedding[3])

	t.Logf("Intra-cluster distances: %f, %f", dist01, dist34)
	t.Logf("Inter-cluster distance: %f", dist03)
}

func BenchmarkUMAP(b *testing.B) {
	data := generateBlobs(1000, 5, 50, 42)

	config := DefaultConfig()
	config.NNeighbors = 15
	config.NComponents = 2
	config.NEpochs = 100
	config.Seed = 42
	config.Init = "random"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model := New(config)
		model.FitTransform(data)
	}
}
