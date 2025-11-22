package attention

import (
	"math"

	"github.com/go-portfolio/go-neuro-transformer/internal/matrix"
)

// Механизм внимания
func Attention(Q, K, V matrix.Matrix) matrix.Matrix {
	// Вычисление скалярного произведения Q и K
	QK := matrix.MatrixMul(Q, matrix.Transpose(K))

	// Масштабируем
	dk := float64(len(K[0]))
	scale := 1.0 / math.Sqrt(dk)
	QK = matrix.ScaleMatrix(QK, scale)

	// Применяем softmax
	softmax := matrix.Softmax(QK)

	// Вычисление результата внимания
	return matrix.MatrixMul(softmax, V)
}
