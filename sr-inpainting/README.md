# ЛР3 (усложнённая). SR-inpainting: улучшение фрагмента изображения диффузионной моделью по маске

**Цель:** разобраться в устройстве диффузионных image-to-image моделей.

**Что сделано:**
1. Выбрана предобученная диффузионная SR-модель **FaithDiff** (CVPR 2025) — развитие SUPIR на том же SDXL-бэкбоне.
2. Процесс сэмплинга переписан так, что улучшается только область, заданная бинарной маской в разрешении входа (SR-inpainting).
3. Алгоритм проверен на DIV2K с тремя видами деградаций (bicubic x4, unknown x4, Real-ESRGAN-подобная) и тремя видами маски (все 1, все 0, комбинированная).
4. Качество оценено метриками PSNR, LPIPS, CLIP-IQA — по всему изображению, а также отдельно внутри и вне маски.

```
.
├── README.md             # отчёт
├── requirements.txt      # зависимости с версиями
├── source/
│   ├── config.py           # все параметры эксперимента
│   ├── data.py             # загрузка DIV2K, кропы, деградация Real-ESRGAN
│   ├── masks.py            # генерация/загрузка масок, перенос маски в латент
│   ├── masked_faithdiff.py # загрузка FaithDiff + модифицированный цикл сэмплинга
│   ├── metrics.py          # PSNR / LPIPS / CLIP-IQA (полные и по областям)
│   ├── run_experiments.py  # весь эксперимент -> results/metrics.csv
│   ├── make_report.py      # графики и таблица -> report/
│   └── infer.py            # одно изображение + своя маска
└── report/               # графики и таблицы (генерируются make_report.py)
```

Датасеты и веса в репозиторий не кладутся: скрипты скачивают их при первом запуске (`data/`, `checkpoints/` и `results/` в `.gitignore`).

---

## 1. Теоретическая база

### 1.1 Диффузионные модели (DDPM)

Диффузионная модель задаёт **прямой процесс**: к изображению $x_0$ постепенно добавляется гауссов шум:

$$q(x_t \mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big),\qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I),$$

где $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ монотонно убывает от ≈1 до ≈0 при $t=1\ldots T$ (обычно $T=1000$). При $t=T$ от изображения остаётся почти чистый шум.

Нейросеть $\varepsilon_\theta(x_t, t, c)$ обучается предсказывать добавленный шум (функция потерь $\|\varepsilon-\varepsilon_\theta(x_t,t,c)\|^2$). Такое предсказание шума эквивалентно оценке score-функции $\nabla_x\log p_t(x)$, т.е. направления к более правдоподобным данным. **Обратный процесс** (генерация) начинается с шума $x_T$ и шаг за шагом убирает шум: на каждом шаге оценивается чистое изображение

$$\hat x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}},$$

и строится менее зашумлённое $x_{t-1}$. Сэмплирование можно записать как решение ОДУ/СДУ (probability flow ODE), поэтому вместо 1000 шагов DDPM используют быстрые решатели: DDIM, DPM-Solver, **UniPC** (его используют авторы FaithDiff, 20 шагов).

### 1.2 Латентная диффузия, SDXL, classifier-free guidance

В **латентной диффузии** (Stable Diffusion) процесс идёт в пространстве VAE: энкодер сжимает изображение $H\times W\times3$ в латент $\frac H8\times\frac W8\times4$, а UNet работает уже там. Это в десятки раз дешевле, но VAE-перекодирование не идеально (слегка меняет пиксели). **SDXL** — крупная латентная модель (UNet ≈2.6 млрд параметров, два текстовых энкодера CLIP, нативное разрешение 1024²).

**Classifier-free guidance (CFG):** UNet вызывается с текстовым условием и без него (или с негативным промптом), итог экстраполируется:
$\varepsilon = \varepsilon_{uncond} + s\,(\varepsilon_{cond}-\varepsilon_{uncond})$, где $s$ — guidance scale (у нас 5).

### 1.3 Диффузионное супер-разрешение (Real-ISR)

Задача SR — по LR-изображению $y$ восстановить HR $x$. Задача некорректна: одному LR соответствует много HR. Регрессионные сети (SwinIR, Real-ESRGAN) склонны выдавать «среднее», т.е. замыленное решение. Диффузионные модели вместо этого **сэмплируют** правдоподобное HR из $p(x\mid y)$ и используют сильный генеративный априор text-to-image модели, поэтому рисуют реалистичные текстуры. Цена — возможные «галлюцинации» и низкий PSNR. Это известный **perception–distortion tradeoff** (Blau & Michaeli, 2018): нельзя одновременно минимизировать искажение (PSNR↑) и максимизировать реалистичность (LPIPS↓, no-reference IQA↑).

Типовая схема (StableSR, DiffBIR, SUPIR, FaithDiff): LR-вход бикубически увеличивается до целевого размера, кодируется и подаётся в UNet как дополнительное условие (ControlNet-подобный адаптер). Генерация идёт с шума, но под контролем этого условия.

**SUPIR** (CVPR 2024) — SDXL + обучаемый адаптер (ZeroSFT), restoration-guided sampling (на каждом шаге $\hat x_0$ подтягивается к латенту LQ), подписи от LLaVA и обучение на 20M изображений.

**FaithDiff** (CVPR 2025), которую мы используем, развивает SUPIR:
* не замораживает диффузионный априор, а **совместно дообучает** UNet SDXL и энкодер LQ-признаков;
* вводит **alignment module**: признаки LQ-энкодера выравниваются с латентом шумного состояния $x_t$ и прибавляются к входу UNet (параметр `input_embedding` в коде);
* энкодер LQ инициализирован VAE-энкодером и дообучен убирать деградации.

В результате FaithDiff заметно точнее держит структуру входа (faithfulness), чем SUPIR. Для работы с большими изображениями используется тайловое сэмплирование (TLC): латент режется на перекрывающиеся тайлы 128×128 (1024 px) и склеивается с гауссовыми весами.

### 1.4 Inpainting диффузией без дообучения

**RePaint** (Lugmayr et al., CVPR 2022) и **Blended (Latent) Diffusion** (Avrahami et al., 2022/2023) показали, что предобученную безусловную модель можно заставить дорисовать только часть изображения, если на каждом шаге обратного процесса заменять «известную» область её зашумлённой до того же уровня версией:

$$x_{t-1} \leftarrow m\odot x_{t-1}^{\text{model}} + (1-m)\odot\Big(\sqrt{\bar\alpha_{t-1}}\,x_0^{\text{known}}+\sqrt{1-\bar\alpha_{t-1}}\,\varepsilon\Big).$$

Так модель на каждом шаге «видит» согласованный контекст и генерирует область маски с учётом окружения. В латентном варианте (Blended Latent Diffusion) склейка делается в латенте, а маска уменьшается в 8 раз. Из-за неидеального VAE после декодирования нужна ещё финальная склейка в пикселях.

В нашей задаче (**SR-inpainting**) «известная» область — это вход, увеличенный бикубикой (неулучшенный), а «дорисовываемая» — область маски, которую модель должна улучшить.

### 1.5 Метрики

* **PSNR** = $10\log_{10}(255^2/\text{MSE})$, по Y-каналу (как принято в SR). Измеряет попиксельную точность (distortion). ↑ лучше.
* **LPIPS** (Zhang et al., 2018) — расстояние между глубокими признаками (AlexNet) результата и эталона, близко к восприятию человека. ↓ лучше.
* **CLIP-IQA** (Wang et al., AAAI 2023) — безэталонная (no-reference) оценка качества: косинусная близость CLIP-эмбеддинга картинки к паре промптов «Good photo.» / «Bad photo.». ↑ лучше. Не требует HR.

---

## 2. Описание разработанной системы

### 2.1 Выбор модели

| Модель | Год | Бэкбон | Почему (не) выбрана |
|---|---|---|---|
| StableSR, DiffBIR, PASD | 2023 | SD 2.1 | старые и заметно слабее |
| SUPIR (рекомендована в задании) | CVPR 2024 | SDXL | хороша, но свой фреймворк (sgm), тяжёлый запуск с LLaVA |
| **FaithDiff** | **CVPR 2025** | **SDXL** | развитие SUPIR, точнее по структуре; **официально встроена в diffusers**, поэтому цикл сэмплинга доступен и понятен |
| OSEDiff, TSD-SR, PiSA-SR | 2024–2025 | SD 2.1/3 | **одношаговые**: блендить по шагам нечего, задание теряет смысл |

FaithDiff — многошаговая модель (20 шагов UniPC), поэтому к ней естественно применяется блендинг из п. 1.4. Используются веса `jychen9811/FaithDiff` поверх `SG161222/RealVisXL_V4.0` и VAE `madebyollin/sdxl-vae-fp16-fix`, как в официальном примере.

### 2.2 Алгоритм SR-inpainting

Вход: LR-изображение $y$ ($h\times w$), бинарная маска $M\in\{0,1\}^{h\times w}$ (1 — улучшать), масштаб $s=4$.

1. **Апскейл:** $y_\uparrow$ = бикубика($y$) до $sh\times sw$; маска увеличивается nearest-ом: $M_\uparrow$.
2. **Условие FaithDiff:** $c$ = LQ-энкодер($y_\uparrow$), как в оригинале.
3. **Известная область:** $z_{ref}=\text{VAE.enc}(y_\uparrow)$ (мода распределения).
4. **Латентная маска** $m$: `area`-уменьшение $M_\uparrow$ в 8 раз (доля покрытия латентного пикселя), расширение на `LATENT_DILATE=1` латентный пиксель (max-pool) и лёгкое размытие. Расширение нужно, чтобы шов на границе генерировался моделью: латентный пиксель VAE «видит» больше 8 px.
5. **Старт:** $x_{T}=\sqrt{\bar\alpha_T}z_{ref}+\sqrt{1-\bar\alpha_T}\varepsilon$ (режим `start_point="lr"` авторов). Шум $\varepsilon$ фиксируется и используется на всех шагах для известной области, поэтому её траектория согласована со стартом.
6. **Цикл по шагам** $t_i$ (20 шагов UniPC):
   * латент режется на тайлы (TLC); **тайлы без маски UNet не считает**: для них сразу берётся $q(z_{ref}, t_{i+1})$. При маленькой маске на больших изображениях это даёт ускорение;
   * для тайлов с маской: CFG-вызов UNet с условием $c$, затем шаг решателя (у каждого тайла своё состояние UniPC);
   * тайлы склеиваются гауссовыми весами;
   * **модификация:** $x_{t_{i+1}} \leftarrow m\odot x_{t_{i+1}} + (1-m)\odot q(z_{ref}, t_{i+1})$, где $q(z,t)=\sqrt{\bar\alpha_t}z+\sqrt{1-\bar\alpha_t}\varepsilon$ (через `DDPMScheduler.add_noise` с тем же β-расписанием). На последнем шаге $t=0$ подставляется чистый $z_{ref}$. Параметр `BLEND_END` позволяет отключить блендинг на последних шагах, чтобы модель гармонизировала шов (по умолчанию 1.0, т.е. блендинг на всех шагах).
7. **Декодирование** VAE → $\hat x$; цветокоррекция AdaIN по $y_\uparrow$ (как `adain_color_fix` у авторов).
8. **Финальная склейка в пикселях:** $\text{out}=\tilde M\odot\hat x+(1-\tilde M)\odot y_\uparrow$, где $\tilde M$ — $M_\uparrow$, размытая гауссом (σ=`FEATHER_PX`=8 px). Вне маски результат совпадает с бикубикой бит-в-бит, т.е. VAE ничего не портит.

Особые случаи:
* $M\equiv0$ → модель не вызывается вовсе, результат = $y_\uparrow$;
* $M\equiv1$ → $m\equiv1$, блендинг ничего не меняет, это эквивалентно оригинальному FaithDiff.

Для склейки нужен VP-решатель (DDIM/UniPC/DPM++ с предсказанием ε или v), где $x_t$ имеет вид $\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$. Для Euler-подобных решателей с σ-параметризацией формула другая, поэтому в `load_pipeline` разрешены только `unipc` и `ddim`.

### 2.3 Архитектура

```
LR (h×w) ─┬─ bicubic ×4 ─ y↑ ─┬─ LQ-энкодер FaithDiff ─► c (условие, align-модуль)
          │                    └─ VAE.enc ─► z_ref ──► q(z_ref,t) ──┐
маска M ──┴─ nearest ×4 ─ M↑ ─┬─ area/8 + dilate ─► m ───────────────┤
                               │                                      ▼
            шум ε ─► x_T ─► [ тайлы ─► SDXL-UNet(+c, CFG) ─► UniPC-шаг ─► склейка тайлов ─► блендинг m ] ×20
                               │                                                                  │
                               └─ gauss blur ─► M̃ ───────► out = M̃·AdaIN(VAE.dec(x_0)) + (1−M̃)·y↑
```

Модифицированный цикл находится в `source/masked_faithdiff.py::masked_super_resolve`. Это переписанная версия `__call__` из community-пайплайна diffusers `pipeline_faithdiff_stable_diffusion_xl.py` (v0.42.1). Сам пайплайн используется только для загрузки компонентов (UNet с FaithDiff-слоями, VAE, текстовые энкодеры, тайлер `LocalAttention`).

### 2.4 Данные и протокол эксперимента

* **DIV2K validation**, изображения `0801, 0804, 0811, 0829, 0853` (`config.IMAGE_IDS`). Из центра берётся LR-кроп 256×256, HR-кроп 1024×1024, т.е. ровно один тайл SDXL: это экономит время и память.
* Деградации:
  * `bicubic` — официальный DIV2K LR bicubic ×4;
  * `unknown` — официальный DIV2K LR unknown ×4 (track 2, неизвестное ядро);
  * `realesrgan` — генерируется из HR кодом `data.py::realesrgan_degrade`: деградация второго порядка (анизотропный гауссов блюр → ресайз со случайной интерполяцией → гауссов/пуассоновский шум → JPEG, дважды), упрощённая версия из статьи Real-ESRGAN. Сид = номер изображения, поэтому результат воспроизводим.
* Маски (`masks.py::make_mask`), в разрешении LR 256×256: `ones`, `zeros`, `combined` (объединение эллипса и прямоугольника, ~25–40% площади, форма зависит от номера изображения).
* Метрики (`metrics.py`, библиотека pyiqa для LPIPS и CLIP-IQA): по всему изображению, а также `psnr_in/psnr_out` (по пикселям внутри/вне маски) и `lpips_in/clipiqa_in` (по ограничивающему прямоугольнику маски). Для сравнения считается также чистая бикубика (`bicubic_baseline`) с той же комбинированной маской.
* Параметры сэмплинга — как у авторов: 20 шагов UniPC, CFG = 5, `start_point="lr"`, seed = 42, AdaIN-цветокоррекция. Промпт общий (без LLaVA), см. `config.py`.

### 2.5 Системные требования

* NVIDIA GPU с CUDA (код community-пайплайна жёстко использует `cuda`). Рекомендуется ≥ 24 ГБ видеопамяти (RTX 3090/4090, A5000, L4 и т.п.). На картах с 12–16 ГБ используйте флаг `--cpu_offload`: медленнее, но меньше памяти. Можно Colab/Kaggle с T4 16 ГБ + `--cpu_offload`.
* ОЗУ ≥ 32 ГБ (особенно с `--cpu_offload`).
* Диск: ~15 ГБ под веса (RealVisXL fp16 ≈ 7 ГБ, текстовые энкодеры, FaithDiff.bin, VAE, CLIP для метрик) + ~0.5 ГБ DIV2K (HR + 2 LR-архива).
* Python 3.11–3.12, зависимости — `requirements.txt`.

### 2.6 Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # для нужной версии CUDA при необходимости см. pytorch.org

# весь эксперимент (5 изображений × 3 деградации × 3 маски)
python source/run_experiments.py         # добавить --cpu_offload, если мало видеопамяти
python source/make_report.py             # графики и таблица -> report/

# быстрый прогон для проверки
python source/run_experiments.py --images 0801 --degradations bicubic

# своё изображение и своя маска (белое = улучшать, размер как у входа)
python source/infer.py --input img.png --mask mask.png --output out.png
```

`run_experiments.py` дописывает `results/metrics.csv` после каждого прогона и пропускает уже посчитанные комбинации, поэтому прерванный эксперимент можно продолжить.

---

## 3. Результаты работы и тестирования

> ⚠️ **Заполнить после запуска.** Графики и таблица генерируются `make_report.py` в `report/`; таблицу из `report/results_table.md` вставить сюда.

### 3.1 Визуальные результаты

Колонки: вход (бикубика), маска, SR с маской 1 / комбинированной / 0, эталон HR; нижний ряд — зум на границу маски.

![bicubic](report/visual_bicubic_0801.png)
![unknown](report/visual_unknown_0801.png)
![realesrgan](report/visual_realesrgan_0801.png)

### 3.2 Метрики по всему изображению

![metrics](report/metrics_full.png)

### 3.3 Комбинированная маска: внутри и вне маски

![in/out](report/metrics_in_out.png)

### 3.4 Таблица (среднее по 5 изображениям)

<!-- вставить содержимое report/results_table.md -->

| degradation | mask | PSNR | LPIPS | CLIP-IQA | PSNR_in | PSNR_out | LPIPS_in | CLIP-IQA_in | время, с |
|---|---|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … | … | … |

### 3.5 Закономерности

> Ниже — ожидаемые из теории закономерности, которые нужно **подтвердить или опровергнуть** по полученным числам и переписать своими словами.

1. **Маска = 0** даёт ровно бикубический апскейл: максимальный или близкий к максимальному PSNR (особенно на `bicubic`), но худшие LPIPS и CLIP-IQA (картинка мыльная).
2. **Маска = 1** (обычный FaithDiff) — лучшие LPIPS/CLIP-IQA и более низкий PSNR (perception–distortion tradeoff). Сильнее всего выигрыш на `unknown` и `realesrgan`: там бикубика сохраняет шум и JPEG-артефакты, а модель их убирает.
3. **Комбинированная маска:** внутри маски метрики близки к варианту «маска = 1», вне маски PSNR совпадает с бикубикой (за счёт склейки в пикселях). Метрики по всему изображению лежат между вариантами 0 и 1, примерно пропорционально площади маски.
4. На границе маски возможен шов (переход «резко → мыльно»). Его видимость регулируют `FEATHER_PX`, `LATENT_DILATE` и `BLEND_END<1`.
5. CLIP-IQA по всему изображению с частичной маской может «штрафовать» неоднородность резкости.

---

## 4. Выводы

> ⚠️ Переписать по фактическим результатам.

* Разобрано устройство латентной диффузионной SR-модели (SDXL + LQ-энкодер + alignment-модуль FaithDiff) и её цикла сэмплинга: CFG, многошаговый решатель UniPC, тайловое сэмплирование.
* Предложен и реализован training-free способ SR-inpainting: блендинг латента с зашумлённым латентом бикубического входа на каждом шаге, плюс финальная пиксельная склейка с размытой маской. Тайлы вне маски пропускаются, что ускоряет обработку при маленькой маске.
* Крайние случаи корректны: маска 0 даёт бикубику, маска 1 — оригинальный FaithDiff.
* [по результатам] Количественно: …
* Ограничения: шов на границе маски; дорого (SDXL, 20 шагов × CFG); PSNR у генеративного SR ниже, чем у регрессионных методов; CLIP-IQA плохо подходит для изображений с неоднородным качеством.

---

## 5. Использованные источники

1. Chen J., Pan J., Dong J. *FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution.* CVPR 2025. [arXiv:2411.18824](https://arxiv.org/abs/2411.18824), [код](https://github.com/JyChen9811/FaithDiff)
2. Пайплайн FaithDiff в diffusers: [pipeline_faithdiff_stable_diffusion_xl.py](https://github.com/huggingface/diffusers/blob/v0.42.1/examples/community/pipeline_faithdiff_stable_diffusion_xl.py)
3. Yu F. et al. *Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild (SUPIR).* CVPR 2024. [arXiv:2401.13627](https://arxiv.org/abs/2401.13627)
4. Ho J., Jain A., Abbeel P. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
5. Song J., Meng C., Ermon S. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
6. Rombach R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
7. Podell D. et al. *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* 2023. [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
8. Ho J., Salimans T. *Classifier-Free Diffusion Guidance.* 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
9. Zhao W. et al. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023. [arXiv:2302.04867](https://arxiv.org/abs/2302.04867)
10. Lugmayr A. et al. *RePaint: Inpainting using Denoising Diffusion Probabilistic Models.* CVPR 2022. [arXiv:2201.09865](https://arxiv.org/abs/2201.09865)
11. Avrahami O., Fried O., Lischinski D. *Blended Latent Diffusion.* SIGGRAPH 2023. [arXiv:2206.02779](https://arxiv.org/abs/2206.02779)
12. Wang X. et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCVW 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
13. Agustsson E., Timofte R. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study (DIV2K).* CVPRW 2017. [сайт датасета](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
14. Zhang R. et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR 2018. [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)
15. Wang J., Chan K.C.K., Loy C.C. *Exploring CLIP for Assessing the Look and Feel of Images (CLIP-IQA).* AAAI 2023. [arXiv:2207.12396](https://arxiv.org/abs/2207.12396)
16. Blau Y., Michaeli T. *The Perception-Distortion Tradeoff.* CVPR 2018. [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
17. Chen C. et al. *pyiqa / IQA-PyTorch.* [github.com/chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
18. Holderrieth P., Erives E. *An Introduction to Flow Matching and Diffusion Models* (MIT 6.S184). [diffusion.csail.mit.edu](https://diffusion.csail.mit.edu/)
