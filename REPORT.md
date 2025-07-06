# Задание 1: Сравнение CNN и полносвязных сетей

## 1.1 Сравнение на MNIST

### Полносвязная сеть

![Полносвязная сеть на MNIST](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/%D0%9F%D0%BE%D0%BB%D0%BD%D0%BE%D1%81%D0%B2%D1%8F%D0%B7%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B5%D1%82%D1%8C%20%D0%BD%D0%B0%20MNIST.png)

### Простая CNN

![Простая CNN на MNIST](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/%D0%9F%D1%80%D0%BE%D1%81%D1%82%D0%B0%D1%8F%20CNN%20%D0%BD%D0%B0%20MNIST.png)

### CNN с Residual Block

![CNN с Residual Block на MNIST](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/CNN%20%D1%81%20Residual%20Block%20%D0%BD%D0%B0%20MNIST.png)

### Результат

| |Время обучения|Точность (train)|Точность (validate)|Время инференса|
|-|--------------|----------------|-------------------|---------------|
|Полносвязная сеть|133.09|0.99|0.98|0.001284|
|Простая CNN|161.93|0.99|0.99|0.002050|
|CNN с Residual Block|248.82|1.00|0.99|0.005978|

### Выводы

- Все модели показывают высокую точность
- Полносвязная сеть уже достигает ```98%``` на валидации, но чуть переобучается (```99%``` на тренировочной).
- Простая CNN лучше справляется с обобщением (тренировочная == валидационная == ```99%```).
- CNN с Residual Block достигает идеальной точности на трейне, но валидируется также, как простая CNN: с переобучением.

В итоге лучше всех себя показала CNN с Residual Block, но она обучалась дольше всех и время инференса у нее самое большое

### Количество параметров
| |Параметров|
|-|----------|
|Полносвязная сеть|242 304|
|Простая CNN|5 216|
|CNN с Residual Block|157 994|

---

## 1.1 Сравнение на CIFAR-10

### Полносвязная сеть

#### График

![Полносвязная сеть на CIFAR](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/%D0%9F%D0%BE%D0%BB%D0%BD%D0%BE%D1%81%D0%B2%D1%8F%D0%B7%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B5%D1%82%D1%8C%20%D0%BD%D0%B0%20CIFAR.png)

#### Confusion Matrix

![Confusion Matrix (Полносвязная сеть)](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/Confusion%20Matrix%20(%D0%9F%D0%BE%D0%BB%D0%BD%D0%BE%D1%81%D0%B2%D1%8F%D0%B7%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B5%D1%82%D1%8C).png)

### CNN с Residual блоками

#### График

![CNN с Residual блоками на CIFAR](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/CNN%20%D1%81%20Residual%20Block%20%D0%BD%D0%B0%20CIFAR.png)

#### Confusion Matrix

![Confusion Matrix (CNN с Residual блоками)](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/Confusion%20Matrix%20(CNN%20%D1%81%20Residual%20Block%20%D0%BD%D0%B0%20CIFAR).png)

### CNN с регуляризацией и Residual блоками

#### График

![CNN с регуляризацией и Residual блоками на CIFAR](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/CNN%20%D1%81%20Residual%20Block%20%D0%B8%20%D1%80%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B5%D0%B9%20%D0%BD%D0%B0%20CIFAR.png)

#### Confusion Matrix

![Confusion Matrix (CNN с регуляризацией и Residual блоками)](https://github.com/Ycalk/Lesson4/raw/main/plots/cnn_vs_fc_comparison/Confusion%20Matrix%20(CNN%20%D1%81%20%D1%80%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B5%D0%B9%20%D0%B8%20Residual%20%D0%B1%D0%BB%D0%BE%D0%BA%D0%B0%D0%BC%D0%B8).png)

### Результат

| |Время обучения|Точность (train)|Точность (validate)|Время инференса|
|-|--------------|----------------|-------------------|---------------|
|Полносвязная сеть|199.41|0.49|0.52|0.006294|
|CNN с Residual Block|257.20|0.89|0.80|0.006189|
|CNN с регуляризацией и Residual блоками|272.47|0.87|0.80|0.005464|

### Выводы

#### Полносвязная сеть

Очень низкая точность ```(49–52%)```, выше случайного угадывания (```10``` классов -> ```10%``` случайно). Так происходит потому что модель не учитывает структуру изображения (пространственные взаимосвязи), в итоге она быстрая (обучается за ```199.41```) но бесполезная.

### CNN с Residual Block

Существенно лучше чем полносвязная: ```89%``` на тренировочных, ```80%``` на валидационных. Хорошее обобщение, но есть немного переобучения. Residual блоки помогают при обучении более глубоких сетей, но увеличивают время обучения.

### CNN с регуляризацией и Residual блоками

Очень похожа по точности на обычную CNN с Residual, но чуть меньше переобучилась (точность на валидации осталась прежняя, но точность на тренировочных данных снизилась с ```89%``` -> ```87%```). Также у этой модели самый быстрый инференс.

В итоге, полносвязная сеть не подходит для CIFAR, так как точность всего ```52%```. Residual + регуляризация показывает наилучшее обобщение, даже несмотря на небольшое снижение train accuracy.
