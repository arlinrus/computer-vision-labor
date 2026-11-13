# ЛР3 (усложнённая). SR-inpainting: улучшение фрагмента изображения диффузионной моделью по маске

**Цель:** разобраться в устройстве диффузионных image-to-image моделей.

**Что сделано:**
1. Выбрана предобученная диффузионная SR-модель **FaithDiff** (CVPR 2025) — развитие SUPIR на том же бэкбоне SDXL.
2. Процесс сэмплинга переписан так, что улучшается только область, заданная бинарной маской в разрешении входа (SR-inpainting).
3. Алгоритм проверен на 5 изображениях DIV2K с тремя деградациями (bicubic ×4, unknown ×4, Real-ESRGAN) и тремя масками (все 1, все 0, комбинированная), а также на 7 реальных плохих фотографиях.
4. Качество оценено метриками PSNR, LPIPS, CLIP-IQA; PSNR дополнительно считается отдельно внутри и вне маски.

```
sr-inpainting/
├── README.md            # отчёт
├── requirements.txt     # зависимости
├── config.py            # все параметры
├── model.py             # загрузка FaithDiff
├── masks.py             # маски 0/1
├── sampling.py          # модифицированный цикл сэмплинга (главный файл)
├── data.py              # DIV2K и деградации
├── metrics.py           # PSNR / LPIPS / CLIP-IQA
├── run_experiments.py   # эксперимент на DIV2K -> results/metrics.csv
├── make_report.py       # таблица и графики -> report/
├── demo.py              # эксперимент на фотографиях из photos/ -> results/photos/
├── photos/              # реальные плохие фотографии 256×256
└── report/              # графики и таблица для отчёта
```

Датасет и веса модели в репозиторий не кладутся: скрипты скачивают их при первом запуске (`data/` и `results/` в `.gitignore`).

---

## 1. Теоретическая база

*Раздел будет дописан.*

---

## 2. Описание разработанной системы

### 2.1 Выбор модели

| Модель | Год | Бэкбон | Почему (не) выбрана |
|---|---|---|---|
| StableSR, DiffBIR, PASD | 2023 | SD 2.1 | старее и слабее |
| SUPIR (рекомендована в задании) | CVPR 2024 | SDXL | свой фреймворк (sgm), EDM-сэмплер, тяжёлый запуск с LLaVA |
| **FaithDiff** | **CVPR 2025** | **SDXL** | развитие SUPIR, точнее по структуре; **встроена в diffusers**, поэтому цикл сэмплинга доступен и его легко модифицировать |
| OSEDiff, TSD-SR | 2024–2025 | SD 2.1/3 | **одношаговые**: подменять область по шагам нечего |

Используются веса `jychen9811/FaithDiff` поверх `SG161222/RealVisXL_V4.0` и VAE `madebyollin/sdxl-vae-fp16-fix`, как в официальном примере. Веса модели не меняются.

### 2.2 Алгоритм SR-inpainting

Вход: LR-изображение $y$ (до 256×256), бинарная маска $M$ того же размера (1 — улучшать), масштаб ×4.

1. **Апскейл:** $y_\uparrow$ = бикубика($y$) до 1024×1024; маска растягивается методом ближайшего соседа: $M_\uparrow$. Если $M\equiv0$, модель не вызывается, результат = $y_\uparrow$.
2. **Текст:** промпт и негативный промпт кодируются текстовыми энкодерами SDXL.
3. **Условие FaithDiff:** $c$ = энкодер FaithDiff($y_\uparrow$), как в оригинале.
4. **Известная область:** $z_{ref}$ = VAE.encode($y_\uparrow$), латент 128×128. **Латентная маска** $m$ = max-pool $M_\uparrow$ с окном 8: латентный пиксель равен 1, если в его квадрате 8×8 есть хоть одна единица маски. Маска в латенте получается чуть шире, и шов генерируется моделью.
5. **Старт:** $x_T=\sqrt{\bar\alpha_T}z_{ref}+\sqrt{1-\bar\alpha_T}\varepsilon$ (режим `start_point="lr"` авторов), шум $\varepsilon$ с фиксированным seed.
6. **Цикл, 20 шагов UniPC:**
   * UNet с условием $c$ и CFG предсказывает шум, решатель делает шаг (как у авторов);
   * **модификация:** $x_{t_{i+1}} \leftarrow m\odot x_{t_{i+1}} + (1-m)\odot q(z_{ref}, t_{i+1})$, где $q(z,t)=\sqrt{\bar\alpha_t}z+\sqrt{1-\bar\alpha_t}\varepsilon$. На последнем шаге подставляется чистый $z_{ref}$.
7. **Декодирование:** VAE.decode → $\hat x$; цветокоррекция AdaIN по $y_\uparrow$ (как у авторов).
8. **Финальная склейка в пикселях:** $\text{out}=\tilde M\odot\hat x+(1-\tilde M)\odot y_\uparrow$, где $\tilde M$ — маска $M_\uparrow$, размытая гауссом (σ = 8 px). Вне маски результат совпадает с бикубикой.

Особые случаи: $M\equiv0$ — модель не вызывается; $M\equiv1$ — подмена ничего не меняет, получается оригинальный FaithDiff.

**Упрощение:** вход ограничен 256×256, поэтому выход 1024×1024 — ровно один тайл SDXL, и латент не нужно резать на перекрывающиеся тайлы, как в оригинальном пайплайне.

### 2.3 Архитектура

```
LR (256) ─┬─ bicubic ×4 ─ y↑ (1024) ─┬─ энкодер FaithDiff ─► c (условие)
          │                          └─ VAE.encode ─► z_ref (128) ──► q(z_ref, t) ──┐
маска M ──┴─ nearest ×4 ─ M↑ ─┬─ max-pool /8 ─► m (128) ────────────────────────────┤
                              │                                                      ▼
           шум ε ─► x_T ─► [ UNet(c, текст, CFG) ─► шаг UniPC ─► m·x + (1−m)·q(z_ref, t) ] × 20
                              │                                                      │
                              └─ gauss blur ─► M̃ ──► out = M̃·AdaIN(VAE.decode(x_0)) + (1−M̃)·y↑
```

| Файл | Роль |
|---|---|
| `config.py` | все параметры: адреса моделей, 20 шагов, CFG 5, seed 42, промпты, картинки, маски |
| `model.py` | `load_model()`: скачивает и собирает FaithDiff, включает экономию памяти, ставит UniPC |
| `masks.py` | `make_mask()` (ones / zeros / combined / half), `load_mask()` из PNG |
| `sampling.py` | `masked_sr()`: модифицированный цикл сэмплинга (алгоритм 2.2) |
| `data.py` | `get_pair()`: LR 256 + HR 1024 из DIV2K; `realesrgan_degrade()` |
| `metrics.py` | PSNR (весь кадр, внутри и вне маски), LPIPS, CLIP-IQA |
| `run_experiments.py` | DIV2K: картинки × деградации × маски → `results/metrics.csv` |
| `make_report.py` | таблица средних и графики → `report/` |
| `demo.py` | фотографии из `photos/` × 4 маски, CLIP-IQA и сравнения → `results/photos/` |

Цикл в `sampling.py` — переписанная версия `__call__` из пайплайна diffusers `pipeline_faithdiff_stable_diffusion_xl.py` (v0.42.1); сам пайплайн используется для загрузки компонентов.

### 2.4 Данные и протокол эксперимента

**Эксперимент 1 — DIV2K (с эталоном, все три метрики).**
* DIV2K validation, изображения `0801, 0804, 0811, 0829, 0853`. Из центра берётся LR-кроп 256×256 и соответствующий HR-кроп 1024×1024.
* Деградации:
  * `bicubic` — официальный DIV2K LR bicubic ×4;
  * `unknown` — официальный DIV2K LR unknown ×4 (неизвестное ядро размытия);
  * `realesrgan` — генерируется из HR в `data.py`: гауссово размытие (σ 1–2.5) → уменьшение ×4 → гауссов шум (σ 5–15) → JPEG (качество 30–60). Упрощённая версия деградации из статьи Real-ESRGAN; seed = номер изображения.
* Маски 256×256: `ones`, `zeros`, `combined` (эллипс + прямоугольник, ~30% площади).
* Метрики: PSNR, LPIPS, CLIP-IQA по всему изображению; `psnr_in` / `psnr_out` — по пикселям внутри и вне маски. Маска `zeros` совпадает с бикубикой и служит базовой линией.

**Эксперимент 2 — реальные фотографии (без эталона).**
* 7 фотографий 256×256 с реальной (неизвестной) деградацией: bears, butterfly, dog, flowers, frog, kingfisher, tiger.
* Маски `ones`, `zeros`, `combined` и дополнительно `half` (левая половина) — чтобы «до» и «после» были видны на одной картинке.
* Эталона нет, поэтому считается только безэталонная метрика CLIP-IQA.

Параметры сэмплинга — значения авторов (`test_wo_llava.py` и пример в diffusers): 20 шагов UniPC, CFG 5, `start_point="lr"`, seed 42, AdaIN. Промпт общий для всех картинок, так как LLaVA не используется.

### 2.5 Системные требования

* NVIDIA GPU с CUDA. Запуск выполнялся в **Kaggle** на Tesla T4 (16 ГБ) с выгрузкой частей модели в ОЗУ (`enable_model_cpu_offload`).
* ОЗУ ≥ 30 ГБ (столько даёт Kaggle).
* Диск: ~15 ГБ под веса модели + ~0.5 ГБ DIV2K.
* Время: ~24 с на одно изображение 1024×1024 (20 шагов) на T4.
* Python 3.11+, зависимости — `requirements.txt` (torch, numpy, Pillow уже есть в Kaggle).

### 2.6 Запуск

*Раздел будет дописан.*

---

## 3. Результаты работы и тестирования

### 3.1 DIV2K: визуальные результаты

Колонки: вход (бикубика), маска, результат с масками ones / zeros / combined, эталон. Нижний ряд — увеличенный центр.

![bicubic](p/visual_bicubic.png)
![unknown](report/visual_unknown.png)
![realesrgan](report/visual_realesrgan.png)

### 3.2 DIV2K: метрики

![metrics](report/metrics.png)

Среднее по 5 изображениям. `nan` — пустая область: у маски ones нет пикселей вне маски, у zeros — внутри.

| Деградация | Маска | PSNR ↑ | LPIPS ↓ | CLIP-IQA ↑ | PSNR внутри | PSNR вне |
|---|---|---|---|---|---|---|
| bicubic | zeros (= бикубика) | **27.95** | 0.435 | 0.478 | — | 27.95 |
| bicubic | combined | 27.92 | 0.410 | 0.459 | 27.25 | 28.32 |
| bicubic | ones | 27.12 | **0.266** | **0.559** | 27.12 | — |
| unknown | zeros (= бикубика) | **22.85** | 0.578 | 0.347 | — | 22.85 |
| unknown | combined | 22.83 | 0.567 | 0.321 | 22.49 | 23.09 |
| unknown | ones | 22.38 | **0.439** | **0.368** | 22.38 | — |
| realesrgan | zeros (= бикубика) | 25.53 | 0.728 | 0.271 | — | 25.53 |
| realesrgan | combined | **25.59** | 0.673 | 0.275 | 25.16 | 25.84 |
| realesrgan | ones | 24.82 | **0.375** | **0.498** | 24.82 | — |

Для маски zeros «PSNR вне» посчитан по всему изображению, а для combined — только по области вне маски, поэтому эти два столбца напрямую не сравниваются.

### 3.3 DIV2K: закономерности

1. **Маска «все 0» даёт ровно бикубику:** модель не вызывается, время ≈ 0 с. Крайний случай обработан корректно.

2. **Маска «все 1» (обычный FaithDiff) — perception–distortion tradeoff.** По сравнению с бикубикой:

   | Деградация | ΔPSNR, дБ | ΔLPIPS | ΔCLIP-IQA |
   |---|---|---|---|
   | bicubic | −0.83 | −39 % | +17 % |
   | unknown | −0.47 | −24 % | +6 % |
   | realesrgan | −0.71 | −48 % | +84 % |

   Перцептивное качество (LPIPS, CLIP-IQA) заметно растёт, а попиксельная точность (PSNR) немного падает: модель дорисовывает правдоподобные текстуры, которые не совпадают с эталоном пиксель в пиксель.

3. **Сильнее всего модель помогает на деградации Real-ESRGAN** (размытие + шум + JPEG): LPIPS падает почти вдвое, CLIP-IQA растёт на 84 %. Бикубика сохраняет шум и блоки JPEG, а FaithDiff обучалась как раз на таких деградациях и убирает их. **Слабее всего — на `unknown`**: неизвестное ядро размытия DIV2K отличается от обучающих деградаций модели.

4. **Комбинированная маска меняет только часть изображения.** Метрики по всему кадру у combined почти совпадают с бикубикой (PSNR 27.92 против 27.95, 22.83 против 22.85), а LPIPS улучшается немного — маска закрывает около 30 % площади. На Real-ESRGAN у combined PSNR даже выше бикубики (25.59 против 25.53): удаление шума внутри маски приближает картинку к эталону.

5. **CLIP-IQA «штрафует» частичное улучшение.** На bicubic и unknown у комбинированной маски CLIP-IQA ниже, чем у чистой бикубики. Картинка с неоднородной резкостью (часть чёткая, часть мыльная) кажется CLIP хуже, чем однородная. Это ограничение безэталонной метрики, которая оценивает изображение целиком.

### 3.4 Реальные фотографии

Сверху: бикубика и результаты с масками ones / combined / half, жёлтая линия — граница маски. Снизу — увеличенный центр.

![dog](photos/dog_grid.png)
![bears](photos/bears_grid.png)
![tiger](photos/tiger_grid.png)

Остальные: `report/photos/*_grid.png`.

CLIP-IQA ↑ (эталона нет, поэтому PSNR и LPIPS не считаются):

| Фото | бикубика | zeros | combined | half | ones | ones vs бикубика |
|---|---|---|---|---|---|---|
| bears | 0.446 | 0.446 | 0.414 | 0.443 | **0.697** | +56 % |
| butterfly | 0.462 | 0.462 | 0.456 | 0.438 | **0.514** | +11 % |
| dog | 0.442 | 0.442 | 0.396 | 0.382 | **0.631** | +43 % |
| flowers | 0.554 | 0.554 | 0.450 | 0.482 | **0.691** | +25 % |
| frog | **0.443** | 0.443 | 0.415 | 0.435 | 0.398 | −10 % |
| kingfisher | **0.417** | 0.417 | 0.381 | 0.377 | 0.301 | −28 % |
| tiger | **0.725** | 0.725 | 0.696 | 0.676 | 0.556 | −23 % |
| **среднее** | 0.498 | 0.498 | 0.458 | 0.462 | **0.541** | +9 % |

### 3.5 Реальные фотографии: закономерности

1. **Маска «все 0» снова в точности равна бикубике** на всех 7 фото.
2. **Сильнее всего модель улучшает шумные фотографии**: dog (+43 %), bears (+56 %), flowers (+25 %). На dog цветной шум убран полностью и появилась текстура шерсти; с маской half граница «улучшено / нет» видна по вертикальной линии.
3. **На слабо испорченных фото (tiger, kingfisher, frog) CLIP-IQA падает.** Эти фотографии в основном размыты и пережаты, шума почти нет. Модель убирает артефакты JPEG, но заодно сглаживает мелкие высокочастотные детали, а CLIP-IQA оценивает резкие контрастные текстуры выше. Визуально результат при этом чище, но «мягче».
4. **Частичные маски (combined, half) дают CLIP-IQA ниже, чем у бикубики** почти на всех фото — та же закономерность, что и на DIV2K: неоднородная резкость оценивается метрикой хуже.

---

## 4. Выводы

* Разобрано устройство латентной диффузионной SR-модели FaithDiff (SDXL + энкодер плохого изображения + alignment-модуль) и её цикла сэмплинга: classifier-free guidance, решатель UniPC на 20 шагов, старт с зашумлённого входа.
* Реализован способ SR-inpainting без дообучения модели: на каждом шаге сэмплинга латент вне маски заменяется латентом бикубического входа, зашумлённым до текущего уровня шума (идея RePaint / Blended Latent Diffusion); после декодирования выполняется склейка в пикселях с размытой границей.
* Крайние случаи корректны: маска «все 0» даёт ровно бикубику без вызова модели, маска «все 1» — обычный FaithDiff.
* С комбинированной маской изменяется только выбранная область: метрики по всему кадру остаются близки к бикубике, а на картинках улучшение видно строго внутри границы маски.
* Подтверждён perception–distortion tradeoff: полное улучшение снижает LPIPS на 24–48 % и повышает CLIP-IQA на 6–84 % ценой падения PSNR на 0.5–0.8 дБ.
* Эффект зависит от типа деградации: максимальный — на шуме и JPEG (Real-ESRGAN, шумные реальные фото), минимальный — на неизвестном размытии и слабо испорченных фото, где модель сглаживает детали.
* Ограничения: CLIP-IQA плохо подходит для оценки частично улучшенных изображений; вход ограничен 256×256 (один тайл); модель тяжёлая (~24 с на изображение 1024×1024 на T4) и требует GPU.

---

## 5. Использованные источники

1. Chen J., Pan J., Dong J. *FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution.* CVPR 2025. [arXiv:2411.18824](https://arxiv.org/abs/2411.18824), [код](https://github.com/JyChen9811/FaithDiff)
2. Пайплайн FaithDiff в diffusers: [pipeline_faithdiff_stable_diffusion_xl.py](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_faithdiff_stable_diffusion_xl.py)
3. Yu F. et al. *Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild (SUPIR).* CVPR 2024. [arXiv:2401.13627](https://arxiv.org/abs/2401.13627)
4. Ho J., Jain A., Abbeel P. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
5. Rombach R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
6. Podell D. et al. *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* 2023. [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
7. Ho J., Salimans T. *Classifier-Free Diffusion Guidance.* 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
8. Zhao W. et al. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023. [arXiv:2302.04867](https://arxiv.org/abs/2302.04867)
9. Lugmayr A. et al. *RePaint: Inpainting using Denoising Diffusion Probabilistic Models.* CVPR 2022. [arXiv:2201.09865](https://arxiv.org/abs/2201.09865)
10. Avrahami O., Fried O., Lischinski D. *Blended Latent Diffusion.* SIGGRAPH 2023. [arXiv:2206.02779](https://arxiv.org/abs/2206.02779)
11. Wang X. et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCVW 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
12. Agustsson E., Timofte R. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study (DIV2K).* CVPRW 2017. [сайт датасета](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
13. Zhang R. et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR 2018. [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)
14. Wang J., Chan K.C.K., Loy C.C. *Exploring CLIP for Assessing the Look and Feel of Images (CLIP-IQA).* AAAI 2023. [arXiv:2207.12396](https://arxiv.org/abs/2207.12396)
15. Blau Y., Michaeli T. *The Perception-Distortion Tradeoff.* CVPR 2018. [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
16. Chen C. et al. *pyiqa / IQA-PyTorch.* [github.com/chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
17. Тестовые фотографии: RealSRSet ([репозиторий PASD](https://github.com/yangxy/PASD)) и демо-набор [DiffBIR](https://github.com/XPixelGroup/DiffBIR).
