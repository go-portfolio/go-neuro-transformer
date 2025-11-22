package transformer

import (
	"github.com/go-portfolio/go-neuro-transformer/internal/attention"
	"github.com/go-portfolio/go-neuro-transformer/internal/layers"
	"github.com/go-portfolio/go-neuro-transformer/internal/matrix"
)

// Трансформер
func Transformer(Q, K, V, W1, W2 matrix.Matrix, b1, b2 []float64) matrix.Matrix {
	// Применяем механизм внимания
	attentionOutput := attention.Attention(Q, K, V)

	// Применяем полносвязный слой
	output := layers.FeedForward(attentionOutput, W1, W2, b1, b2)

	return output
}
