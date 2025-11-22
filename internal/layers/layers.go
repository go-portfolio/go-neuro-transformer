package layers

import "github.com/go-portfolio/go-neuro-transformer/internal/matrix"

// Функция активации ReLU
func ReLU(x matrix.Matrix) matrix.Matrix {
	for i := range x {
		for j := range x[i] {
			if x[i][j] < 0 {
				x[i][j] = 0
			}
		}
	}
	return x
}

// Полносвязный слой (Feedforward)
func FeedForward(x matrix.Matrix, W1, W2 matrix.Matrix, b1, b2 []float64) matrix.Matrix {
	// Применяем первый линейный слой
	x1 := matrix.MatrixMul(x, W1)
	x1 = AddBias(x1, b1)
	x1 = ReLU(x1)

	// Применяем второй линейный слой
	x2 := matrix.MatrixMul(x1, W2)
	x2 = AddBias(x2, b2)

	return x2
}

// Добавление смещения
func AddBias(x matrix.Matrix, b []float64) matrix.Matrix {
	for i := range x {
		for j := range x[i] {
			x[i][j] += b[j]
		}
	}
	return x
}
