# Image classification

<div align="center">
    <a href="">
    <img width="400" height="280" src="pic.png">
    </a>
    <h3>Image classification</h4>
</div>

----

## Как обучить модель:

1. Подготавливаем датасет:
    1. Скачиваем датасет с изображениями из инструмента разметки, подготавливаем разбиения.
    2. Оформляем структуру:

        ```bash
       <DATASET_DIR>
        ├── train
            ├── <class_name_1>
                ├── <file_name_1>
            ├── <class_name_1>
            └── ...
        └── validation
        ```

2. В директории `configs/` создаём конфиг `<task_name>.yml`. Он полностью наследует `base.yml`, но имеет смысл
   переопределить некоторые его части:
    1. `experiment`

       ```yml
        experiment:
          name: "<your_exp_name>"
          work_dir: "</your/exp/dir>"
        ```

       Выбираем название эксперимента, задаём `seed` и `work_dir`. В `work_dir` создастся следующая структура:

        ```bash
        <work_dir>/
            └──<exp_name>/
                └──%d.%m/
                    ├──%H.%M.%S/
                    ├──...
        ```

    2. `training.augs`:

        ```yml
        training:
          augs:
            level: <low/high> # пока что пара режимов с разной степенью аугов
        ```

    3. `training`

        ```yml
        dataset_dir: <dataset_dir>
        ```

    4. `model`
        Параметры модели.
        Указываем название модели. Подойдут любые работающие с `timm`
Теперь можно учить:

```bash
make train CONFIG=<task_name>.yml
```
