package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

// Матрица
type Matrix [][]float64

// Умножение матриц
func MatrixMul(a, b Matrix) Matrix {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if colsA != rowsB {
		log.Fatal("Invalid matrix dimensions")
	}
	result := make(Matrix, rowsA)
	for i := 0; i < rowsA; i++ {
		result[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// Транспонирование матрицы
func Transpose(a Matrix) Matrix {
	rows, cols := len(a), len(a[0])
	result := make(Matrix, cols)
	for i := 0; i < cols; i++ {
		result[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			result[i][j] = a[j][i]
		}
	}
	return result
}

// Масштабирование матрицы
func ScaleMatrix(a Matrix, scale float64) Matrix {
	rows, cols := len(a), len(a[0])
	result := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] * scale
		}
	}
	return result
}

// Softmax
func Softmax(x Matrix) Matrix {
	rows, cols := len(x), len(x[0])
	result := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
	}
	for i := 0; i < rows; i++ {
		var sumExp float64
		for j := 0; j < cols; j++ {
			result[i][j] = math.Exp(x[i][j])
			sumExp += result[i][j]
		}
		for j := 0; j < cols; j++ {
			result[i][j] /= sumExp
		}
	}
	return result
}

// Механизм внимания
func Attention(Q, K, V Matrix) Matrix {
	// Вычисление скалярного произведения Q и K
	QK := MatrixMul(Q, Transpose(K))

	// Масштабируем
	dk := float64(len(K[0]))
	scale := 1.0 / math.Sqrt(dk)
	QK = ScaleMatrix(QK, scale)

	// Применяем softmax
	softmax := Softmax(QK)

	// Вычисление результата внимания
	return MatrixMul(softmax, V)
}

// Функция активации ReLU
func ReLU(x Matrix) Matrix {
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
func FeedForward(x Matrix, W1, W2 Matrix, b1, b2 []float64) Matrix {
	// Применяем первый линейный слой
	x1 := MatrixMul(x, W1)
	x1 = AddBias(x1, b1)
	x1 = ReLU(x1)

	// Применяем второй линейный слой
	x2 := MatrixMul(x1, W2)
	x2 = AddBias(x2, b2)

	return x2
}

// Добавление смещения
func AddBias(x Matrix, b []float64) Matrix {
	for i := range x {
		for j := range x[i] {
			x[i][j] += b[j]
		}
	}
	return x
}

// Трансформер
func Transformer(Q, K, V, W1, W2 Matrix, b1, b2 []float64) Matrix {
	// Применяем механизм внимания
	attentionOutput := Attention(Q, K, V)

	// Применяем полносвязный слой
	output := FeedForward(attentionOutput, W1, W2, b1, b2)

	return output
}

// Тестирование
func main() {
	rand.Seed(time.Now().UnixNano())

	// Примерные данные
	Q := Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример запросов
	K := Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример ключей
	V := Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример значений

	// Весовые матрицы и смещения для полносвязного слоя
	W1 := Matrix{{0.5, 0.5}, {0.5, 0.5}}
	W2 := Matrix{{0.5, 0.5}, {0.5, 0.5}}
	b1 := []float64{0.1, 0.1}
	b2 := []float64{0.1, 0.1}

	// Получаем результат
	result := Transformer(Q, K, V, W1, W2, b1, b2)

	// Печатаем результат
	fmt.Println("Final Output:", result)
}
