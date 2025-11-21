package attention

import (
	"math"

	"github.com/go-portfolio/go-neuro-transformer/internal/matrix"
)

// Attention рассчитывает механизм внимания (Scaled Dot-Product Attention)
func Attention(Q, K, V [][]float64) [][]float64 {
	// 1. Вычисление скалярного произведения Q и K (dot product)
	QK := matrix.Mul(Q, matrix.Transpose(K))

	// 2. Масштабируем (делим на корень из размерности ключа)
	dk := float64(len(K[0])) // Размерность ключа
	scale := 1.0 / math.Sqrt(dk)
	QK = matrix.Scale(QK, scale)

	// 3. Применяем softmax (будем использовать экспоненциальную функцию для упрощения)
	softmax := make([][]float64, len(QK))
	for i := range QK {
		softmax[i] = make([]float64, len(QK[i]))
		var sumExp float64
		for j := range QK[i] {
			softmax[i][j] = math.Exp(QK[i][j])
			sumExp += softmax[i][j]
		}
		for j := range softmax[i] {
			softmax[i][j] /= sumExp
		}
	}

	// 4. Вычисление внимания: softmax(QK) * V
	return matrix.Mul(softmax, V)
}
