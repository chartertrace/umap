// Package layout provides the optimization layout algorithms for UMAP.
// This implements stochastic gradient descent to optimize the low-dimensional embedding.
package layout

import (
	"unsafe"

	"github.com/nozzle/umap/graph"
	"github.com/nozzle/umap/internal/parallel"
	"github.com/nozzle/umap/internal/rand"
)

// LayoutConfig configures the layout optimization.
type LayoutConfig struct {
	// MinDist is the minimum distance between points in the embedding
	MinDist float32
	// Spread is the effective scale of embedded points
	Spread float32
	// NegativeSampleRate is the ratio of negative samples per positive sample
	NegativeSampleRate int
	// NEpochs is the number of epochs to run
	NEpochs int
	// LearningRate is the initial learning rate (alpha)
	LearningRate float32
	// Seed for random number generation
	Seed int64
	// NumWorkers for parallel processing (0 = auto)
	// Note: When Seed is set, single-threaded mode is used for reproducibility
	NumWorkers int
	// Verbose enables progress output
	Verbose bool
	// ProgressCallback is called after each epoch with (epoch, totalEpochs)
	ProgressCallback func(epoch, total int)
	// RNGState is the Tausworthe RNG state for SGD (optional, for exact reproducibility)
	// If nil, it will be derived from Seed using MT19937
	RNGState []int64
}

// DefaultLayoutConfig returns default configuration.
func DefaultLayoutConfig() LayoutConfig {
	return LayoutConfig{
		MinDist:            0.1,
		Spread:             1.0,
		NegativeSampleRate: 5,
		NEpochs:            200,
		LearningRate:       1.0,
		Seed:               42,
		NumWorkers:         0,
		Verbose:            false,
	}
}

// curveParameters stores the a and b parameters for the curve
// that maps distance in embedding space to membership strength.
type curveParameters struct {
	a, b float32
}

// findABParams finds parameters a, b such that:
// 1 / (1 + a * d^(2b)) approximates a smooth curve with:
// - value 1 at d = 0
// - value ~1 at d <= minDist
// - value ~0 at d = spread
// - approaching 0 as d -> infinity
//
// This matches the Python UMAP implementation which uses scipy.optimize.curve_fit
// (Levenberg-Marquardt algorithm) to find optimal parameters.
func findABParams(minDist, spread float32) curveParameters {
	if minDist <= 0 {
		minDist = 0.001
	}
	if spread <= 0 {
		spread = 1.0
	}

	// The target curve in Python UMAP is:
	// if x < minDist: y = 1.0
	// else: y = exp(-(x - minDist) / spread)
	//
	// We fit: 1 / (1 + a * x^(2*b)) to this curve

	// Sample points for fitting (matching Python's linspace)
	nSamples := 300
	xMax := 3.0 * float64(spread)
	xs := make([]float64, nSamples)
	ys := make([]float64, nSamples)
	for i := range nSamples {
		// Python uses linspace which includes endpoint
		x := float64(i) / float64(nSamples-1) * xMax
		xs[i] = x
		// Target function from Python UMAP
		if x < float64(minDist) {
			ys[i] = 1.0
		} else {
			ys[i] = fastExp(-(x - float64(minDist)) / float64(spread))
		}
	}

	// Use Gauss-Newton / Levenberg-Marquardt style optimization
	// Start with reasonable initial values
	a := float64(1.0)
	b := float64(1.0)

	// Levenberg-Marquardt with adaptive damping
	lambda := 0.001

	for range 1000 {
		// Compute Jacobian and residuals
		var jTj00, jTj01, jTj11 float64 // J^T * J matrix elements
		var jTr0, jTr1 float64          // J^T * r vector elements
		var totalError float64

		for i := range nSamples {
			x := xs[i]
			yTarget := ys[i]

			// Skip x=0 to avoid numerical issues
			if x < 1e-10 {
				continue
			}

			// y = 1 / (1 + a * x^(2b))
			x2b := fastPow(x, 2*b)
			denom := 1.0 + a*x2b
			yPred := 1.0 / denom

			// Residual
			r := yPred - yTarget
			totalError += r * r

			// Jacobian elements
			// dy/da = -x^(2b) / (1 + a*x^(2b))^2
			dydA := -x2b / (denom * denom)

			// dy/db = -2*a*x^(2b)*ln(x) / (1 + a*x^(2b))^2
			lnX := fastLog(x)
			dydB := -2 * a * x2b * lnX / (denom * denom)

			// Accumulate J^T * J and J^T * r
			jTj00 += dydA * dydA
			jTj01 += dydA * dydB
			jTj11 += dydB * dydB
			jTr0 += dydA * r
			jTr1 += dydB * r
		}

		// Solve (J^T*J + lambda*I) * delta = -J^T*r using 2x2 matrix inversion
		m00 := jTj00 + lambda
		m01 := jTj01
		m10 := jTj01
		m11 := jTj11 + lambda

		det := m00*m11 - m01*m10
		if det == 0 || det != det { // Check for zero or NaN
			break
		}

		// Inverse of 2x2 matrix
		inv00 := m11 / det
		inv01 := -m01 / det
		inv10 := -m10 / det
		inv11 := m00 / det

		// Compute update
		deltaA := -(inv00*jTr0 + inv01*jTr1)
		deltaB := -(inv10*jTr0 + inv11*jTr1)

		// Try update
		newA := a + deltaA
		newB := b + deltaB

		// Keep in reasonable range
		if newA < 0.01 {
			newA = 0.01
		}
		if newA > 100 {
			newA = 100
		}
		if newB < 0.1 {
			newB = 0.1
		}
		if newB > 2.0 {
			newB = 2.0
		}

		// Compute new error
		newError := float64(0)
		for i := range nSamples {
			x := xs[i]
			if x < 1e-10 {
				continue
			}
			yTarget := ys[i]
			x2b := fastPow(x, 2*newB)
			yPred := 1.0 / (1.0 + newA*x2b)
			r := yPred - yTarget
			newError += r * r
		}

		if newError < totalError {
			// Accept update, decrease lambda
			a = newA
			b = newB
			lambda *= 0.1
			if lambda < 1e-10 {
				lambda = 1e-10
			}
		} else {
			// Reject update, increase lambda
			lambda *= 10
			if lambda > 1e10 {
				break
			}
		}

		// Check convergence
		if totalError < 1e-12 {
			break
		}
	}

	return curveParameters{a: float32(a), b: float32(b)}
}

// fastExp computes exp(x) quickly for optimization
func fastExp(x float64) float64 {
	if x > 88 {
		return 1e38
	}
	if x < -88 {
		return 0
	}
	// Use Taylor series
	sum := 1.0
	term := 1.0
	for i := 1; i < 30; i++ {
		term *= x / float64(i)
		sum += term
		if term < 1e-15 && term > -1e-15 {
			break
		}
	}
	return sum
}

// fastPow computes x^y quickly for optimization
func fastPow(x, y float64) float64 {
	if x <= 0 {
		return 0
	}
	return fastExp(y * fastLog(x))
}

// fastLog computes ln(x) quickly for optimization
func fastLog(x float64) float64 {
	if x <= 0 {
		return -1e38
	}
	// Use the identity: ln(x) = 2 * arctanh((x-1)/(x+1))
	// Which converges as: 2 * sum_{n=0}^inf (1/(2n+1)) * ((x-1)/(x+1))^(2n+1)
	y := (x - 1) / (x + 1)
	y2 := y * y
	sum := y
	term := y
	for i := 1; i < 100; i++ {
		term *= y2
		sum += term / float64(2*i+1)
		if term < 1e-15 && term > -1e-15 {
			break
		}
	}
	return 2 * sum
}

// OptimizeLayout runs the SGD optimization loop.
func OptimizeLayout(
	embedding [][]float32,
	g *graph.CSRMatrix,
	config LayoutConfig,
) {
	n := len(embedding)
	if n == 0 {
		return
	}
	dim := len(embedding[0])

	// For reproducibility with a seed, use single-threaded mode
	// This matches Python UMAP behavior when random_state is set
	useParallel := config.Seed == 0 && config.NumWorkers != 1

	// Find curve parameters
	params := findABParams(config.MinDist, config.Spread)
	a, b := params.a, params.b

	// Compute epochs per sample for each edge
	epochsPerSample := graph.ToEpochsPerSample(g, config.NEpochs)

	// Get edges
	heads, tails, _ := g.GetEdges()
	nEdges := len(heads)

	// Track when each edge should next be sampled
	epochsPerNegSample := make([]float32, nEdges)
	epochOfNextSample := make([]float32, nEdges)
	epochOfNextNegSample := make([]float32, nEdges)

	for i := range nEdges {
		epochsPerNegSample[i] = epochsPerSample[i] / float32(config.NegativeSampleRate)
		epochOfNextSample[i] = epochsPerSample[i]
		epochOfNextNegSample[i] = epochsPerNegSample[i]
	}

	// Initialize base random state
	var baseRNG rand.State
	if len(config.RNGState) >= 3 {
		baseRNG = rand.State{config.RNGState[0], config.RNGState[1], config.RNGState[2]}
	} else {
		baseRNG = rand.New(config.Seed)
	}

	// Create per-vertex RNG states (matching Python UMAP's rng_state_per_sample)
	// Python does: rng_state_per_sample = base_rng_state + embedding[:, 0].view(int64)
	rngStatePerSample := make([]rand.State, n)
	for i := range n {
		// Convert float32 -> float64 -> int64 (matching Python's .view(np.int64))
		f64 := float64(embedding[i][0])
		i64 := float64ToInt64Bits(f64)

		rngStatePerSample[i] = rand.State{
			baseRNG[0] + i64,
			baseRNG[1] + i64,
			baseRNG[2] + i64,
		}
	}

	// Optimization loop
	alpha := config.LearningRate
	moveOther := float32(1.0) // Python UMAP's fit() uses move_other=True

	for epoch := 0; epoch < config.NEpochs; epoch++ {
		if useParallel {
			// Parallel processing (non-reproducible)
			numWorkers := config.NumWorkers
			if numWorkers <= 0 {
				numWorkers = parallel.NumWorkers()
			}
			parallel.ParallelFor(0, nEdges, numWorkers, func(edge int) {
				processEdgePython(embedding, heads, tails, epochsPerSample, epochsPerNegSample,
					epochOfNextSample, epochOfNextNegSample, edge, epoch, n, dim, a, b, alpha, moveOther, rngStatePerSample)
			})
		} else {
			// Sequential processing (reproducible)
			for edge := range nEdges {
				processEdgePython(embedding, heads, tails, epochsPerSample, epochsPerNegSample,
					epochOfNextSample, epochOfNextNegSample, edge, epoch, n, dim, a, b, alpha, moveOther, rngStatePerSample)
			}
		}

		// Decay learning rate
		alpha = config.LearningRate * (1.0 - float32(epoch)/float32(config.NEpochs))
		if alpha < 0.0001 {
			alpha = 0.0001
		}

		// Progress callback
		if config.ProgressCallback != nil {
			config.ProgressCallback(epoch+1, config.NEpochs)
		}
	}
}

// float64ToInt64Bits reinterprets a float64 as int64 (bit-level cast).
// This matches Python's: np.float64(x).view(np.int64)
func float64ToInt64Bits(f float64) int64 {
	return *(*int64)(unsafe.Pointer(&f))
}

// processEdgePython processes a single edge during SGD optimization.
// This matches the Python UMAP implementation exactly.
func processEdgePython(
	embedding [][]float32,
	heads, tails []int32,
	epochsPerSample, epochsPerNegSample, epochOfNextSample, epochOfNextNegSample []float32,
	edge, epoch, n, dim int,
	a, b, alpha, moveOther float32,
	rngStatePerSample []rand.State,
) {
	// Check if this edge should be sampled this epoch
	if epochOfNextSample[edge] > float32(epoch) {
		return
	}

	j := int(heads[edge])
	k := int(tails[edge])

	// Apply attractive force
	current := embedding[j]
	other := embedding[k]

	// Compute squared distance
	var distSq float32
	for d := range dim {
		diff := current[d] - other[d]
		distSq += diff * diff
	}

	// Attractive gradient
	var gradCoef float32
	if distSq > 0.0 {
		gradCoef = (-2.0 * a * b * pow32(distSq, b-1.0)) / (a*pow32(distSq, b) + 1.0)
	}

	for d := range dim {
		gradD := clip(gradCoef * (current[d] - other[d]))
		current[d] += gradD * alpha
		if moveOther > 0 {
			other[d] -= gradD * alpha
		}
	}

	// Update next sample time
	epochOfNextSample[edge] += epochsPerSample[edge]

	// Calculate number of negative samples (matching Python exactly)
	nNegSamples := max(int((float32(epoch)-epochOfNextNegSample[edge])/epochsPerNegSample[edge]), 0)

	// Perform negative sampling using the head vertex's RNG state
	rng := &rngStatePerSample[j]
	gamma := float32(1.0) // repulsion weight

	for p := 0; p < nNegSamples; p++ {
		// Sample random negative vertex
		// Python's modulo wraps negative numbers: -5 % 15 = 10
		// Go's modulo preserves sign: -5 % 15 = -5
		// We need to match Python's behavior
		randVal := rand.Int(rng)
		kNeg := int(randVal) % n
		if kNeg < 0 {
			kNeg += n
		}

		otherNeg := embedding[kNeg]

		// Compute squared distance
		var distSqNeg float32
		for d := range dim {
			diff := current[d] - otherNeg[d]
			distSqNeg += diff * diff
		}

		// Repulsive gradient
		var gradCoefNeg float32
		if distSqNeg > 0.0 {
			gradCoefNeg = (2.0 * gamma * b) / ((0.001 + distSqNeg) * (a*pow32(distSqNeg, b) + 1.0))
		} else if j == kNeg {
			continue
		}

		for d := range dim {
			var gradD float32
			if gradCoefNeg > 0.0 {
				gradD = clip(gradCoefNeg * (current[d] - otherNeg[d]))
			}
			current[d] += gradD * alpha
		}
	}

	// Update next negative sample time
	epochOfNextNegSample[edge] += float32(nNegSamples) * epochsPerNegSample[edge]
}

// clip clamps a value to [-4, 4] range.
func clip(val float32) float32 {
	if val > 4.0 {
		return 4.0
	}
	if val < -4.0 {
		return -4.0
	}
	return val
}

// pow32 computes x^y for float32.
func pow32(x, y float32) float32 {
	// Fast path for common exponents
	if y == 1 {
		return x
	}
	if y == 0 {
		return 1
	}
	if y == 0.5 {
		return sqrt32(x)
	}
	if y == 2 {
		return x * x
	}

	// General case using exp/log
	if x <= 0 {
		return 0
	}
	return exp32(y * log32(x))
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x
	for range 10 {
		z = (z + x/z) / 2
	}
	return z
}

func exp32(x float32) float32 {
	// Limit input range
	if x > 88 {
		return 1e38
	}
	if x < -88 {
		return 0
	}

	// Taylor series approximation
	sum := float32(1.0)
	term := float32(1.0)
	for i := 1; i < 20; i++ {
		term *= x / float32(i)
		sum += term
		if term < 1e-10 && term > -1e-10 {
			break
		}
	}
	return sum
}

func log32(x float32) float32 {
	if x <= 0 {
		return -1e38
	}
	// Use Newton's method for ln
	y := (x - 1) / (x + 1)
	y2 := y * y
	sum := y
	term := y
	for i := 1; i < 50; i++ {
		term *= y2
		sum += term / float32(2*i+1)
	}
	return 2 * sum
}
