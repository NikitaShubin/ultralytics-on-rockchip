#!/usr/bin/env bash
# Стресс-тест Orange Pi (RK3588S): CPU + NPU + RAM + температура.
# Ctrl+C корректно останавливает всю нагрузку и завершает скрипт.
#
# Использование (обычный стресс-тест):
#   ./stress_test.sh                          # всё, до Ctrl+C
#   ./stress_test.sh -d 600                   # всё, 10 минут
#   ./stress_test.sh --npu-only               # только NPU
#   ./stress_test.sh --cpu-only               # только CPU+RAM
#   ./stress_test.sh -c 0,1 -s 4              # 2 ядра × 4 потока NPU
#   ./stress_test.sh -m yolo11n_none.rknn     # другая модель
#   ./stress_test.sh -l results.csv           # лог в CSV
#
# Режим разгона (только с --overclock):
#   ./stress_test.sh --overclock --oc-setup   # DTB модификация → перезагрузка
#   ./stress_test.sh --overclock              # пошаговый разгон CPU+NPU+GPU
#   ./stress_test.sh --overclock --oc-step 300    # 5 мин на шаг
#   ./stress_test.sh --overclock --oc-restore     # откат DTB → перезагрузка

set -uo pipefail

# ─── Аргументы ─────────────────────────────────────────────────────────────

DURATION=0
NPU_ONLY=0
CPU_ONLY=0
NO_RAM=0
NPU_CORES="0,1,2"
NPU_STREAMS=1
MODEL="model/model.rknn"
INTERVAL=2
LOG_FILE=""
RAM_MB=1024

# ─── Overclock ──────────────────────────────────────────────────────────────

OVERCLOCK=0
OC_SETUP=0
OC_RESTORE=0
OC_STEP=300
OC_TARGETS="cpu,npu,gpu"
OC_REPORT=""

usage() {
    cat <<EOF
Использование: $(basename "$0") [опции]

Обычный стресс-тест:
  -d, --duration SEC     Длительность в секундах (0 = до Ctrl+C, по умолчанию)
  -c, --npu-cores LIST   NPU ядра через запятую (по умолчанию: 0,1,2)
  -s, --npu-streams N    Потоков инференса на ядро (по умолчанию: 8)
  -m, --model FILE       RKNN модель (по умолчанию: model/model.rknn)
  -i, --interval SEC     Интервал мониторинга (по умолчанию: 2)
  -l, --log FILE         CSV лог результатов
  --npu-only             Только NPU
  --cpu-only             Только CPU + RAM
  --no-ram               Без нагрузки на RAM

Режим разгона (все флаги требуют --overclock):
  --overclock            Пошаговый разгон CPU + NPU + GPU
  --oc-setup             DTB модификация (разблокировать 2352/2400 МГц) и перезагрузить
  --oc-restore           Откат DTB к оригиналу и перезагрузить
  --oc-step SEC          Длительность одного шага (по умолчанию: 300)
  --oc-targets LIST      Что разгонять: cpu,npu,gpu (по умолчанию: всё)
  --oc-report FILE       CSV-отчёт с результатами шагов

  -h, --help             Помощь
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--duration)    DURATION="$2"; shift 2 ;;
        -c|--npu-cores)   NPU_CORES="$2"; shift 2 ;;
        -s|--npu-streams) NPU_STREAMS="$2"; shift 2 ;;
        -m|--model)       MODEL="$2"; shift 2 ;;
        -i|--interval)    INTERVAL="$2"; shift 2 ;;
        -l|--log)         LOG_FILE="$2"; shift 2 ;;
        --npu-only)       NPU_ONLY=1; shift ;;
        --cpu-only)       CPU_ONLY=1; shift ;;
        --no-ram)         NO_RAM=1; shift ;;
        --overclock)      OVERCLOCK=1; shift ;;
        --oc-setup)       OC_SETUP=1; shift ;;
        --oc-restore)     OC_RESTORE=1; shift ;;
        --oc-step)        OC_STEP="$2"; shift 2 ;;
        --oc-targets)     OC_TARGETS="$2"; shift 2 ;;
        --oc-report)      OC_REPORT="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

# ─── Валидация overclock флагов ─────────────────────────────────────────────

if [[ -n "$OC_REPORT" || "$OC_TARGETS" != "cpu,npu,gpu" || $OC_STEP -ne 300 ]]; then
    if [[ $OVERCLOCK -eq 0 ]]; then
        echo "[ОШИБКА] Флаги --oc-step, --oc-targets, --oc-report"
        echo "         требуют --overclock. Пример:"
        echo "         ./stress_test.sh --overclock --oc-step 300"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/$MODEL"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ОШИБКА] Модель не найдена: $MODEL_PATH"
    exit 1
fi

# ─── Overclock: константы ───────────────────────────────────────────────────

freq_to_mhz() {
    # Конвертирует частоту в МГц (авто-определение единиц: Hz или kHz)
    local freq=$1
    if [[ $freq -ge 1000000000 ]]; then
        echo $((freq / 1000000))
    elif [[ $freq -ge 1000000 ]]; then
        echo $((freq / 1000))
    else
        echo "$freq"
    fi
}

freq_display() {
    # Возвращает строку вида "2304 МГц" для вывода
    local freq=$1
    local mhz
    mhz=$(freq_to_mhz "$freq")
    echo "${mhz} МГц"
}

OC_DTB="/boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb"
OC_DTB_BAK="/boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb.bak"
OC_DTS="/tmp/oc_base.dts"

# Частотные шаги (фильтруются по доступным на устройстве)
OC_CPU_STEPS=(1800000 2016000 2208000 2256000 2304000 2352000 2400000)
OC_NPU_STEPS=(600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)
OC_GPU_STEPS=(400000000 500000000 600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)

# Thermal throttle частоты (порог троттлинга)
OC_CPU_THROTTLE=2208000
OC_NPU_THROTTLE=800000000
OC_GPU_THROTTLE=800000000

# Результаты разгона (компонент → макс. стабильная частота)
declare -A OC_MAX_STABLE=()

# ─── Overclock: sudo проверка ───────────────────────────────────────────────

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "[ОШИБКА] Требуются права root. Запустите:"
        echo "         sudo $(basename "$0") $*"
        exit 1
    fi
}

# ─── Overclock: проверка DTB ──────────────────────────────────────────────────

is_oc_installed() {
    # Проверяет, что 2352000 есть в scaling_available_frequencies на policy6
    cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "2352000"
}

is_oc_backup_exists() {
    [[ -f "$OC_DTB_BAK" ]]
}

# ─── Overclock: установка DTB модификации ────────────────────────────────────

oc_setup() {
    require_root "$@"
    echo "════════════════════════════════════════════════════════"
    echo "  УСТАНОВКА OVERCLOCK (DTB модификация)"
    echo "════════════════════════════════════════════════════════"

    # Проверить, не установлен ли уже
    if is_oc_installed; then
        echo "[OK] Overclock уже установлен (2352 МГц доступен в CPU frequencies)"
        return 0
    fi

    # Бэкап оригинального DTB
    if [[ ! -f "$OC_DTB_BAK" ]]; then
        cp "$OC_DTB" "$OC_DTB_BAK"
        echo "[OK] Бэкап DTB: $OC_DTB_BAK"
    fi

    # Декомпилировать DTB → DTS
    dtc -I dtb -O dts -o "$OC_DTS" "$OC_DTB" 2>/dev/null
    if [[ ! -f "$OC_DTS" ]]; then
        echo "[ОШИБКА] Не удалось декомпилировать DTB"
        exit 1
    fi
    echo "[OK] DTB декомпилирован: $OC_DTS"

    # Модифицировать DTS: расблокировать opp-supported-hw для CPU
    sed -i \
        '/opp-2352000000/,/};/ {
            /opp-supported-hw/s/=.*/= <0xffffffff 0xffffffff>;/
        }' "$OC_DTS"
    sed -i \
        '/opp-2400000000/,/};/ {
            /opp-supported-hw/s/=.*/= <0xffffffff 0xffffffff>;/
        }' "$OC_DTS"

    # Модифицировать DTS: добавить 1050/1100 MHz OPPs для NPU (после opp-1000000000)
    NPU_1050_OPP='
				opp-1050000000 {
					opp-supported-hw = <0xf9 0xffffffff>;
					opp-hz = <0x00 0x3e919e40>;
					opp-microvolt = <0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0>;
				};
				opp-1100000000 {
					opp-supported-hw = <0xf9 0xffffffff>;
					opp-hz = <0x00 0x4190ce80>;
					opp-microvolt = <0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0>;
				};'

    # Вставляем после NPU opp-1000000000 (в двух кластерах NPU)
    awk -v ins="$NPU_1050_OPP" '
        /opp-1000000000 \{/ { found=1 }
        found && /^\t\t\t\};/ { print; print ins; found=0; next }
        { print }
    ' "$OC_DTS" > /tmp/oc_base_fixed.dts && mv /tmp/oc_base_fixed.dts "$OC_DTS"

    # Добавить 1050/1100 MHz OPPs для GPU
    GPU_1050_OPP='
				opp-1050000000 {
					opp-supported-hw = <0xf9 0xffffffff>;
					opp-hz = <0x00 0x3e919e40>;
					opp-microvolt = <0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0>;
				};
				opp-1100000000 {
					opp-supported-hw = <0xf9 0xffffffff>;
					opp-hz = <0x00 0x4190ce80>;
					opp-microvolt = <0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0>;
				};'

    awk -v ins="$GPU_1050_OPP" '
        /opp-1000000000 \{/ { found=1 }
        found && /^\t\t\t\};/ { print; print ins; found=0; next }
        { print }
    ' "$OC_DTS" > /tmp/oc_base_fixed.dts && mv /tmp/oc_base_fixed.dts "$OC_DTS"

    # Скомпилировать обратно DTB
    dtc -I dts -O dtb -o /tmp/oc_new.dtb "$OC_DTS" 2>/dev/null
    if [[ ! -s /tmp/oc_new.dtb ]]; then
        echo "[ОШИБКА] Не удалось скомпилировать DTB"
        exit 1
    fi

    # Установить новый DTB
    cp /tmp/oc_new.dtb "$OC_DTB"
    echo "[OK] DTB модификация установлена: $OC_DTB"

    echo ""
    echo "Для применения изменений необходима перезагрузка."
    read -r -p "Перезагрузить сейчас? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        reboot
    else
        echo "Не забудьте перезагрузиться: sudo reboot"
    fi
}

# ─── Overclock: откат DTB модификации ────────────────────────────────────────

oc_restore() {
    require_root "$@"
    echo "════════════════════════════════════════════════════════"
    echo "  ОТКАТ OVERCLOCK (восстановление DTB)"
    echo "════════════════════════════════════════════════════════"

    if is_oc_backup_exists; then
        cp "$OC_DTB_BAK" "$OC_DTB"
        echo "[OK] Оригинальный DTB восстановлен из бэкапа"
    else
        echo "[ОШИБКА] Бэкап DTB не найден: $OC_DTB_BAK"
        exit 1
    fi

    # Восстановить governors
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        echo "ondemand" > "$policy/scaling_governor" 2>/dev/null || true
    done
    echo "rknpu_ondemand" > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || true
    echo "simple_ondemand" > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null || true
    echo "[OK] Governors восстановлены (ondemand)"

    echo ""
    echo "Для применения изменений необходима перезагрузка."
    read -r -p "Перезагрузить сейчас? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        reboot
    else
        echo "Не забудьте перезагрузиться: sudo reboot"
    fi
}

# ─── Overclock: управление частотой ─────────────────────────────────────────

set_cpu_freq() {
    local freq=$1
    for policy in /sys/devices/system/cpu/cpufreq/policy{4,6}; do
        [[ -d "$policy" ]] || continue
        echo "userspace" > "$policy/scaling_governor" 2>/dev/null || true
        echo "$freq" > "$policy/scaling_setspeed" 2>/dev/null || true
    done
}

restore_cpu_governor() {
    for policy in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do
        [[ -d "$policy" ]] || continue
        echo "ondemand" > "$policy/scaling_governor" 2>/dev/null || true
    done
}

set_cpu_max_freq() {
    local freq=$1
    for policy in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do
        [[ -d "$policy" ]] || continue
        echo "$freq" > "$policy/scaling_max_freq" 2>/dev/null || true
    done
}

get_cpu_cur_freq() {
    cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq 2>/dev/null || echo 0
}

set_npu_freq() {
    local freq=$1
    echo "userspace" > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/min_freq 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/max_freq 2>/dev/null || true
}

restore_npu_governor() {
    echo "rknpu_ondemand" > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || true
}

set_npu_max_freq() {
    local freq=$1
    echo "userspace" > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/min_freq 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/max_freq 2>/dev/null || true
}

get_npu_cur_freq() {
    cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || echo 0
}

set_gpu_freq() {
    local freq=$1
    echo "userspace" > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/min_freq 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/max_freq 2>/dev/null || true
}

restore_gpu_governor() {
    echo "simple_ondemand" > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null || true
}

set_gpu_max_freq() {
    local freq=$1
    echo "userspace" > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/min_freq 2>/dev/null || true
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/max_freq 2>/dev/null || true
}

get_gpu_cur_freq() {
    cat /sys/class/devfreq/fb000000.gpu/cur_freq 2>/dev/null || echo 0
}

# ─── Overclock: фильтрация доступных частот ─────────────────────────────────

filter_available_freqs() {
    # $1 = "cpu"|"npu"|"gpu"
    # Возвращает только те частоты из OC_*_STEPS, что есть в available_frequencies
    local component=$1
    local available=""
    case "$component" in
        cpu)
            available=$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null)
            ;;
        npu)
            available=$(cat /sys/class/devfreq/fdab0000.npu/available_frequencies 2>/dev/null)
            ;;
        gpu)
            available=$(cat /sys/class/devfreq/fb000000.gpu/available_frequencies 2>/dev/null)
            ;;
    esac

    local var_name="OC_${component^^}_STEPS"
    eval "local steps=(\"\${${var_name}[@]}\")"

    local result=()
    for freq in "${steps[@]}"; do
        if echo "$available" | grep -qw "$freq"; then
            result+=("$freq")
        fi
    done
    echo "${result[@]}"
}

# ─── Overclock: детекция проблем ────────────────────────────────────────────

get_max_temp() {
    local max=0
    for f in "${THERMAL_FILES[@]}"; do
        local t
        t=$(cat "$f" 2>/dev/null || echo 0)
        t=$((t / 1000))
        [[ $t -gt $max ]] && max=$t
    done
    echo "$max"
}

get_temp_by_name() {
    local name=$1
    for i in "${!THERMAL_NAMES[@]}"; do
        if [[ "${THERMAL_NAMES[$i]}" == "$name" ]]; then
            cat "${THERMAL_FILES[$i]}" 2>/dev/null | awk '{printf "%d", $1/1000}'
            return
        fi
    done
    echo 0
}

check_dmesg_errors() {
    local before=$1
    dmesg 2>/dev/null | tail -n 50 | grep -ci "error\|fail\|panic\|oops\|rk3588" | head -1
}

check_stability() {
    # $1 = компонент, $2 = частота, $3 = dmesg count до теста
    # Возвращает: "PASS" или "FAIL_REASON"
    local component=$1
    local freq=$2
    local dmesg_before=$3

    # Проверка 1: dmesg ошибки
    local dmesg_after
    dmesg_after=$(dmesg 2>/dev/null | tail -n 50 | grep -ci "error\|fail\|panic\|oops\|rk3588" | head -1)
    dmesg_after=${dmesg_after:-0}
    if [[ $dmesg_after -gt ${dmesg_before:-0} ]]; then
        echo "FAIL_DMESG"
        return
    fi

    # Проверка 2: thermal throttling
    local throttle_freq=""
    case "$component" in
        cpu) throttle_freq=$OC_CPU_THROTTLE ;;
        npu) throttle_freq=$OC_NPU_THROTTLE ;;
        gpu) throttle_freq=$OC_GPU_THROTTLE ;;
    esac
    local cur_freq
    case "$component" in
        cpu) cur_freq=$(get_cpu_cur_freq) ;;
        npu) cur_freq=$(get_npu_cur_freq) ;;
        gpu) cur_freq=$(get_gpu_cur_freq) ;;
    esac
    if [[ "$cur_freq" -le "$throttle_freq" && "$freq" -gt "$throttle_freq" ]]; then
        echo "FAIL_THROTTLE"
        return
    fi

    # Проверка 3: максимальная температура
    local max_temp
    max_temp=$(get_max_temp)
    if [[ $max_temp -gt 110 ]]; then
        echo "FAIL_TEMP_${max_temp}"
        return
    fi

    # Проверка 4: NPU процесс жив (если тестируем NPU)
    if [[ "$component" == "npu" && -n "$NPU_PID" ]]; then
        if ! kill -0 "$NPU_PID" 2>/dev/null; then
            echo "FAIL_NPU_CRASH"
            return
        fi
    fi

    echo "PASS"
}

# ─── Overclock: мониторинг одного шага ──────────────────────────────────────

monitor_step() {
    local elapsed=0
    local duration=$1
    while true; do
        sleep "$INTERVAL"
        [[ $STOP -eq 1 ]] && return
        elapsed=$((elapsed + INTERVAL))
        [[ $elapsed -ge $duration ]] && return

        local ts
        ts=$(date +%H:%M:%S)
        local cpu_pct; cpu_pct=$(read_cpu_load)
        local npu_pct; npu_pct=$(npu_load_avg)

        printf "  %-8s %4s%% %4s%%" "$ts" "$cpu_pct" "$npu_pct"
        for f in "${THERMAL_FILES[@]}"; do
            local raw
            raw=$(cat "$f" 2>/dev/null || echo "0")
            printf " %4s" "$((raw / 1000))"
        done
        echo ""
    done
}

# ─── Overclock: основной цикл ──────────────────────────────────────────────

overclock_loop() {
    require_root "$@"

    # Проверить DTB модификацию
    if ! is_oc_installed; then
        echo "[ВНИМАНИЕ] Overclock DTB не обнаружен (2352/2400 МГц недоступны)."
        echo "           Разгон будет работать с доступными частотами."
        echo ""
    fi

    echo "════════════════════════════════════════════════════════"
    echo "  ПОШАГОВЫЙ РАЗГОН"
    echo "════════════════════════════════════════════════════════"
    echo "  Шаг: ${OC_STEP}с | Цели: $OC_TARGETS"
    echo "════════════════════════════════════════════════════════"
    echo ""

    # CSV-отчёт
    if [[ -n "$OC_REPORT" ]]; then
        echo "component,freq_hz,step_sec,temp_max,temp_bigcore,temp_npu,temp_gpu,cpu_pct,npu_pct,status" > "$OC_REPORT"
    fi

    # Определяем компоненты для разгона
    IFS=',' read -ra targets <<< "$OC_TARGETS"

    for component in "${targets[@]}"; do
        component=$(echo "$component" | tr -d ' ')
        case "$component" in
            cpu|npu|gpu) ;;
            *) echo "[ПРЕДУПРЕЖДЕНИЕ] Неизвестный компонент: $component, пропуск"; continue ;;
        esac

        # Фильтруем доступные частоты
        read -ra avail_freqs <<< "$(filter_available_freqs "$component")"
        if [[ ${#avail_freqs[@]} -eq 0 ]]; then
            echo "[ПРЕДУПРЕЖДЕНИЕ] Нет доступных частот для $component, пропуск"
            continue
        fi

        echo "─── $component: ${avail_freqs[*]} ───"
        echo ""

        local max_stable=0

        for freq in "${avail_freqs[@]}"; do
            [[ $STOP -eq 1 ]] && break

            echo "  ┌─ $component → $(freq_display "$freq") (${OC_STEP}с)"

            # Установить частоту
            case "$component" in
                cpu) set_cpu_freq "$freq" ;;
                npu) set_npu_freq "$freq" ;;
                gpu) set_gpu_freq "$freq" ;;
            esac

            # Дать системе стабилизироваться
            sleep 3

            # Запомнить dmesg count до теста
            local dmesg_before
            dmesg_before=$(dmesg 2>/dev/null | tail -n 50 | grep -ci "error\|fail\|panic\|oops\|rk3588" | head -1)
            dmesg_before=${dmesg_before:-0}

            # Запустить стресс (если ещё не запущен)
            if [[ ${#STRESS_PIDS[@]} -eq 0 ]]; then
                if [[ $NPU_ONLY -eq 0 ]]; then
                    start_cpu_stress
                    [[ $NO_RAM -eq 0 ]] && start_ram_stress
                fi
                if [[ $CPU_ONLY -eq 0 ]]; then
                    start_npu_stress
                    sleep 5
                fi
            fi

            # Мониторинг
            monitor_step "$OC_STEP"

            # Остановить нагрузку для этого шага
            for pid in "${STRESS_PIDS[@]}"; do
                kill -TERM "$pid" 2>/dev/null || true
            done
            pkill -f 'stress --cpu' 2>/dev/null || true
            pkill -f 'stress --vm' 2>/dev/null || true
            STRESS_PIDS=()

            if [[ -n "$NPU_PID" ]] && kill -0 "$NPU_PID" 2>/dev/null; then
                kill -INT "$NPU_PID" 2>/dev/null || true
                sleep 2
                kill -9 "$NPU_PID" 2>/dev/null || true
                NPU_PID=""
            fi

            # Проверить стабильность
            local result
            result=$(check_stability "$component" "$freq" "$dmesg_before")

            local temp_max; temp_max=$(get_max_temp)
            local temp_big; temp_big=$(get_temp_by_name "bigcore0-thermal")
            local temp_npu; temp_npu=$(get_temp_by_name "npu-thermal")
            local temp_gpu; temp_gpu=$(get_temp_by_name "gpu-thermal")
            local cpu_pct; cpu_pct=$(read_cpu_load)
            local npu_pct; npu_pct=$(npu_load_avg)

            if [[ "$result" == "PASS" ]]; then
                echo "  └─ PASS (t_max=${temp_max}°C, cpu=${cpu_pct}%, npu=${npu_pct}%)"
                max_stable=$freq
            else
                echo "  └─ FAIL: $result (t_max=${temp_max}°C)"
                break
            fi

            # CSV
            if [[ -n "$OC_REPORT" ]]; then
                echo "${component},${freq},${OC_STEP},${temp_max},${temp_big},${temp_npu},${temp_gpu},${cpu_pct},${npu_pct},${result}" >> "$OC_REPORT"
            fi

            echo ""
        done

        OC_MAX_STABLE[$component]=$max_stable
        echo "  ✓ $component: макс. стабильная = $(freq_to_mhz "$max_stable") МГц"
        echo ""
    done

    # ─── Финал: восстановить governors, выставить макс. частоты ────────────

    echo "════════════════════════════════════════════════════════"
    echo "  ЗАВЕРШЕНИЕ"
    echo "════════════════════════════════════════════════════════"

    # Восстановить governors
    restore_cpu_governor
    restore_npu_governor
    restore_gpu_governor

    # Выставить макс. стабильные частоты
    for component in "${!OC_MAX_STABLE[@]}"; do
        local freq=${OC_MAX_STABLE[$component]}
        if [[ $freq -gt 0 ]]; then
            case "$component" in
                cpu) set_cpu_max_freq "$freq" ;;
                npu) set_npu_max_freq "$freq" ;;
                gpu) set_gpu_max_freq "$freq" ;;
            esac
            echo "[OK] $component: governor=ondemand, max=$(freq_display "$freq")"
        fi
    done

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  РЕЗУЛЬТАТЫ РАЗГОНА"
    echo "════════════════════════════════════════════════════════"
    echo ""
    printf "  %-12s %-12s %-18s %s\n" "КОМПОНЕНТ" "ШТАТНЫЙ" "МАКС. СТАБИЛЬНЫЙ" "ИЗМЕНЕНИЕ"
    printf "  %-12s %-12s %-18s %s\n" "──────────" "──────────" "────────────────" "────────"

    local defaults=("cpu:2304000" "npu:1000000000" "gpu:1000000000")
    for entry in "${defaults[@]}"; do
        local comp=${entry%%:*}
        local def=${entry#*:}
        local stable=${OC_MAX_STABLE[$comp]:-0}
        if [[ $stable -gt 0 ]]; then
            local def_mhz=$(freq_to_mhz "$def")
            local stable_mhz=$(freq_to_mhz "$stable")
            local pct=$(( (stable_mhz - def_mhz) * 100 / def_mhz ))
            printf "  %-12s %-10s МГц %-16s МГц +${pct}%%\n" "$comp" "$def_mhz" "$stable_mhz"
        fi
    done

    echo ""
    echo "  Конфигурация оставлена разогнанной."
    echo "  Для отката: sudo $(basename "$0") --overclock --oc-restore"
    echo "  Для полного отката DTB: sudo cp $OC_DTB_BAK $OC_DTB && sudo reboot"
    echo "════════════════════════════════════════════════════════"

    [[ -n "$OC_REPORT" ]] && echo "  CSV-отчёт: $OC_REPORT"
}

# ─── Термозоны ──────────────────────────────────────────────────────────────

declare -a THERMAL_NAMES=()
declare -a THERMAL_FILES=()
declare -a THERMAL_SHORT=()

thermal_short_name() {
    case "$1" in
        soc-thermal)        echo "SOC" ;;
        bigcore0-thermal)   echo "BIG0" ;;
        bigcore1-thermal)   echo "BIG1" ;;
        littlecore-thermal) echo "LIT" ;;
        center-thermal)     echo "CTR" ;;
        gpu-thermal)        echo "GPU" ;;
        npu-thermal)        echo "NPU" ;;
        *) echo "${1%%-thermal}" | cut -c1-5 ;;
    esac
}

init_thermal() {
    local i=0
    for zone_dir in /sys/class/thermal/thermal_zone*; do
        [[ -f "$zone_dir/type" && -f "$zone_dir/temp" ]] || continue
        local name
        name=$(cat "$zone_dir/type")
        THERMAL_NAMES[$i]="$name"
        THERMAL_FILES[$i]="$zone_dir/temp"
        THERMAL_SHORT[$i]=$(thermal_short_name "$name")
        ((i++))
    done
}

get_temps() {
    local result=""
    for i in "${!THERMAL_NAMES[@]}"; do
        local raw
        raw=$(cat "${THERMAL_FILES[$i]}" 2>/dev/null || echo "0")
        local temp=$((raw / 1000))
        result+="${THERMAL_NAMES[$i]}=${temp}° "
    done
    echo "$result"
}

# ─── CPU load ───────────────────────────────────────────────────────────────

CPU_STATE_FILE="/tmp/.stress_cpu_state"

read_cpu_load() {
    # Читает /proc/stat, считает дельту между вызовами.
    # Состояние хранится в файле (обход subshell-проблемы bash $(...))
    local vals
    read -ra vals < <(head -1 /proc/stat)
    local user=${vals[1]:-0}
    local nice=${vals[2]:-0}
    local system=${vals[3]:-0}
    local idle=${vals[4]:-0}
    local iowait=${vals[5]:-0}
    local irq=${vals[6]:-0}
    local softirq=${vals[7]:-0}
    local steal=${vals[8]:-0}

    local cur_idle=$((idle + iowait))
    local cur_total=$((user + nice + system + idle + iowait + irq + softirq + steal))

    if [[ ! -f "$CPU_STATE_FILE" ]]; then
        echo "${cur_idle} ${cur_total}" > "$CPU_STATE_FILE"
        echo 0
        return
    fi

    local prev_idle prev_total
    read -r prev_idle prev_total < "$CPU_STATE_FILE"
    echo "${cur_idle} ${cur_total}" > "$CPU_STATE_FILE"

    local d_idle=$((cur_idle - prev_idle))
    local d_total=$((cur_total - prev_total))

    if [[ $d_total -eq 0 ]]; then
        echo 0
        return
    fi

    echo $(( (d_total - d_idle) * 100 / d_total ))
}

# ─── NPU load ───────────────────────────────────────────────────────────────

read_npu_load_cores() {
    local raw
    raw=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo "")
    if [[ -z "$raw" ]]; then
        echo ""
        return
    fi
    echo "$raw" | grep -oP 'Core\d+:\s*\d+%' | sed 's/Core/C/g;s/: /=/' | tr '\n' ' '
}

npu_load_avg() {
    local raw
    raw=$(read_npu_load_cores)
    if [[ -n "$raw" ]]; then
        local sum=0 cnt=0
        for token in $raw; do
            local v="${token#*=}"
            v="${v%%%}"
            if [[ "$v" =~ ^[0-9]+$ ]]; then
                sum=$((sum + v))
                cnt=$((cnt + 1))
            fi
        done
        if [[ $cnt -gt 0 ]]; then
            echo $((sum / cnt))
            return
        fi
    fi
    # Fallback: /sys/class/devfreq (формат "100@1000000000Hz")
    local devfreq
    devfreq=$(cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null || echo "")
    if [[ -n "$devfreq" ]]; then
        local pct="${devfreq%%@*}"
        if [[ "$pct" =~ ^[0-9]+$ ]]; then
            echo "$pct"
            return
        fi
    fi
    echo 0
}

# ─── RAM ────────────────────────────────────────────────────────────────────

read_ram() {
    awk '/^MemTotal:/{total=$2} /^MemAvailable:/{avail=$2} END{printf "%d %d", total/1024, (total-avail)/1024}' /proc/meminfo
}

# ─── Управление процессами ─────────────────────────────────────────────────

STRESS_PIDS=()
NPU_PID=""
NPU_SCRIPT=""
STOP=0
CLEANUP_DONE=0

kill_npu_stress_in_container() {
    # Контейнер не содержит pkill/kill — используем Python для поиска и убийства процессов
    docker exec percept python3 -c "
import os, signal, glob
for f in glob.glob('/proc/[0-9]*/cmdline'):
    try:
        pid = int(f.split('/')[2])
        if pid <= 1:
            continue
        cmd = open(f, 'rb').read().decode('utf-8', errors='replace')
        if 'npu_stress' in cmd:
            os.kill(pid, signal.SIGKILL)
    except (OSError, ValueError, PermissionError):
        pass
" 2>/dev/null || true
}

cleanup() {
    [[ $CLEANUP_DONE -eq 1 ]] && return
    CLEANUP_DONE=1

    echo ""
    echo "[ОСТАНОВКА] Завершаю все процессы..."

    # Убить stress
    for pid in "${STRESS_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    pkill -f 'stress --cpu' 2>/dev/null || true
    pkill -f 'stress --vm' 2>/dev/null || true

    # Убить NPU Python процессы внутри контейнера (pkill недоступен → Python)
    kill_npu_stress_in_container
    # Также убить хост-процесс docker exec (если ещё жив)
    if [[ -n "$NPU_PID" ]] && kill -0 "$NPU_PID" 2>/dev/null; then
        kill -9 "$NPU_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    [[ -n "$NPU_SCRIPT" ]] && rm -f "$NPU_SCRIPT"
    rm -f "$CPU_STATE_FILE"

    echo "[ГОТОВО] Нагрузка снята"
}

on_signal() {
    STOP=1
    # Передать сигнал NPU Python процессу, чтобы он остановил потоки
    [[ -n "$NPU_PID" ]] && kill -INT "$NPU_PID" 2>/dev/null || true
}

trap on_signal INT TERM
trap 'cleanup; exit 0' EXIT

# ─── CPU/RAM stress ────────────────────────────────────────────────────────

start_cpu_stress() {
    local n_workers
    n_workers=$(nproc)
    nice -n 19 stress --cpu "$n_workers" --timeout 999d &>/dev/null &
    STRESS_PIDS+=($!)
    echo "[CPU] stress: $n_workers воркеров (PID $!)"
}

start_ram_stress() {
    local n_workers=$(( $(nproc) / 2 ))
    [[ $n_workers -lt 1 ]] && n_workers=1
    nice -n 19 stress --vm "$n_workers" --vm-bytes "${RAM_MB}M" --vm-keep --timeout 999d &>/dev/null &
    STRESS_PIDS+=($!)
    echo "[RAM] stress: ${n_workers} × ${RAM_MB} МБ (PID $!)"
}

# ─── NPU stress (Python через Docker, rknnlite только в контейнере) ──────────

NPU_CONTAINER_SCRIPT="/tmp/npu_stress_oc.py"

start_npu_stress() {
    local host_script
    host_script=$(mktemp /tmp/npu_stress_XXXXXX.py)
    cat > "$host_script" <<'PYEOF'
import sys, signal, threading, time, numpy as np

CORES_MAP = {
    0: 'NPU_CORE_0', 1: 'NPU_CORE_1', 2: 'NPU_CORE_2',
    3: 'NPU_CORE_0_1', 4: 'NPU_CORE_0_1_2',
}

model_path = sys.argv[1]
core_ids = [int(x) for x in sys.argv[2].split(',')]
streams = int(sys.argv[3])
stop = threading.Event()

def run(core_id, idx):
    from rknnlite.api import RKNNLite
    core_name = CORES_MAP.get(core_id, f'NPU_CORE_{core_id}')
    mask = getattr(RKNNLite, core_name, None)
    if mask is None:
        return
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        print(f'[NPU] core{core_id}/s{idx}: load failed', file=sys.stderr); return
    if rknn.init_runtime(core_mask=mask) != 0:
        print(f'[NPU] core{core_id}/s{idx}: init failed', file=sys.stderr); return
    dummy = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
    print(f'[NPU] core{core_id}/s{idx}: started', flush=True)
    while not stop.is_set():
        try:
            rknn.inference(inputs=[dummy])
        except Exception as e:
            print(f'[NPU] core{core_id}/s{idx}: {e}', file=sys.stderr)
            time.sleep(1)
    rknn.release()
    print(f'[NPU] core{core_id}/s{idx}: stopped', flush=True)

for cid in core_ids:
    for s in range(streams):
        t = threading.Thread(target=run, args=(cid, s), daemon=True)
        t.start()

signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
while True:
    time.sleep(60)
PYEOF
    # Копируем скрипт в контейнер
    docker cp "$host_script" percept:"$NPU_CONTAINER_SCRIPT" 2>/dev/null
    rm -f "$host_script"
    # Маппинг модели: хост → Docker (/app/model/)
    local docker_model="/app/model/$(basename "$MODEL_PATH")"
    # Остановить предыдущий NPU stress в контейнере (если есть)
    kill_npu_stress_in_container
    sleep 1
    # Запуск через docker exec в фоне (хост PID = docker exec процесс)
    docker exec percept python3 "$NPU_CONTAINER_SCRIPT" "$docker_model" "$NPU_CORES" "$NPU_STREAMS" > /dev/null 2>&1 &
    NPU_PID=$!
    echo "[NPU] ядра=$NPU_CORES × $NPU_STREAMS потоков модель=$(basename "$MODEL_PATH") (PID $NPU_PID)"
}

# ─── Мониторинг ─────────────────────────────────────────────────────────────

monitor() {
    local elapsed=0

    [[ -n "$LOG_FILE" ]] && echo "time,elapsed,cpu%,npu%,ram_used_mb,temps" > "$LOG_FILE"

    # Заголовок: строка имён
    printf "%-8s %5s %5s %5s" "ВРЕМЯ" "CPU%" "NPU%" "RAM"
    for short in "${THERMAL_SHORT[@]}"; do
        printf " %4s" "$short"
    done
    echo ""
    # Разделитель (─ = 3 байта UTF-8, printf считает байты)
    printf "%s\n" "───────────────────────────────────────────────────────────"
    echo ""

    while true; do
        sleep "$INTERVAL"
        [[ $STOP -eq 1 ]] && break

        elapsed=$((elapsed + INTERVAL))

        # Проверка таймаута
        if [[ $DURATION -gt 0 && $elapsed -ge $DURATION ]]; then
            break
        fi

        local ts
        ts=$(date +%H:%M:%S)
        local cpu_pct; cpu_pct=$(read_cpu_load)
        local npu_pct; npu_pct=$(npu_load_avg)
        read -r _ ram_used <<< "$(read_ram)"

        printf "%-8s %4s%% %4s%% %5s" "$ts" "$cpu_pct" "$npu_pct" "$ram_used"
        for f in "${THERMAL_FILES[@]}"; do
            local raw
            raw=$(cat "$f" 2>/dev/null || echo "0")
            printf " %4s" "$((raw / 1000))"
        done
        echo ""

        [[ -n "$LOG_FILE" ]] && {
            local temps_csv=""
            for f in "${THERMAL_FILES[@]}"; do
                local raw; raw=$(cat "$f" 2>/dev/null || echo "0")
                [[ -n "$temps_csv" ]] && temps_csv+=","
                temps_csv+="$((raw / 1000))"
            done
            echo "${ts},${elapsed},${cpu_pct},${npu_pct},${ram_used},${temps_csv}" >> "$LOG_FILE"
        }
    done
}

# ─── Результаты ─────────────────────────────────────────────────────────────

print_results() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  РЕЗУЛЬТАТЫ"
    echo "════════════════════════════════════════════════════════"
    echo ""
    for i in "${!THERMAL_NAMES[@]}"; do
        local raw
        raw=$(cat "${THERMAL_FILES[$i]}" 2>/dev/null || echo "0")
        printf "  %-10s %3d°C\n" "${THERMAL_SHORT[$i]}" $((raw / 1000))
    done
    local npu_cores; npu_cores=$(read_npu_load_cores)
    [[ -n "$npu_cores" ]] && echo -e "\nNPU load: $npu_cores"
    local rt ru; read -r rt ru <<< "$(read_ram)"
    echo "RAM: ${ru}/${rt} МБ"
    [[ -n "$LOG_FILE" ]] && echo "Лог: $LOG_FILE"
}

# ─── main ───────────────────────────────────────────────────────────────────

init_thermal

# ─── Overclock: быстрые команды (выход после выполнения) ────────────────────

if [[ $OC_SETUP -eq 1 ]]; then
    oc_setup "$@"
    exit 0
fi

if [[ $OC_RESTORE -eq 1 ]]; then
    oc_restore "$@"
    exit 0
fi

# ─── Overclock: пошаговый разгон ────────────────────────────────────────────

if [[ $OVERCLOCK -eq 1 ]]; then
    echo "════════════════════════════════════════════════════════"
    echo "  РЕЖИМ РАЗГОНА"
    echo "════════════════════════════════════════════════════════"
    echo "  Шаг: ${OC_STEP}с | Цели: $OC_TARGETS"
    [[ -n "$OC_REPORT" ]] && echo "  CSV: $OC_REPORT"
    echo "════════════════════════════════════════════════════════"
    echo ""
    overclock_loop "$@"
    exit 0
fi

# ─── Обычный стресс-тест ────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
if [[ $DURATION -gt 0 ]]; then
    echo "  СТРЕСС-ТЕСТ: ${DURATION}с"
else
    echo "  СТРЕСС-ТЕСТ: до Ctrl+C"
fi
[[ $NPU_ONLY -eq 0 ]] && echo "  CPU: $(nproc) ядер (stress, nice=19)"
[[ $CPU_ONLY -eq 0 ]] && echo "  NPU: $NPU_CORES × $NPU_STREAMS потоков"
echo "  Модель: $MODEL"
echo "════════════════════════════════════════════════════════"
echo "════════════════════════════════════════════════════════"
echo ""

if [[ $NPU_ONLY -eq 0 ]]; then
    start_cpu_stress
    [[ $NO_RAM -eq 0 ]] && start_ram_stress
fi

if [[ $CPU_ONLY -eq 0 ]]; then
    start_npu_stress
    sleep 5
fi

monitor

print_results
