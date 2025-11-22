package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-neuro-transformer/internal/matrix"
	"github.com/go-portfolio/go-neuro-transformer/internal/transformer"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Примерные данные
	Q := matrix.Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример запросов
	K := matrix.Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример ключей
	V := matrix.Matrix{{1.0, 0.0}, {0.0, 1.0}} // Пример значений

	// Весовые матрицы и смещения для полносвязного слоя
	W1 := matrix.Matrix{{0.5, 0.5}, {0.5, 0.5}}
	W2 := matrix.Matrix{{0.5, 0.5}, {0.5, 0.5}}
	b1 := []float64{0.1, 0.1}
	b2 := []float64{0.1, 0.1}

	// Получаем результат
	result := transformer.Transformer(Q, K, V, W1, W2, b1, b2)

	// Печатаем результат
	fmt.Println("Финальный вывод:", result)
}
