#!/usr/bin/env bash
# Менеджер разгона с проверкой стабильности.
# Определяет нестабильность по падению FPS и автоматически откатывает.
#
# Использование:
#   ./oc_manager.sh status              — текущее состояние
#   ./oc_manager.sh test [--quick]      — тест всех частот, поиск макс. стабильной
#   ./oc_manager.sh apply               — применить сохранённую конфигурацию
#   ./oc_manager.sh revert              — откат к стоковым частотам
#   ./oc_manager.sh monitor             — мониторинг + автооткат при нестабильности
#   ./oc_manager.sh set cpu=2400 npu=1000 gpu=800  — установить конкретные частоты
set -uo pipefail
cd "$(dirname "$0")"

CONTAINER="percept"
OC_DIR="$(dirname "$0")/overclock"
OC_STATE="$OC_DIR/oc_state.conf"
OC_DTB="/boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb"
OC_DTB_BAK="/boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb.bak"
OC_DTS="/tmp/oc_base.dts"

# Модель: аргумент → env var → текущая развёрнутая → yolo11n
detect_model() {
    local name_file="${MODEL_DIR:-model}/model_name.txt"
    if [[ -f "$name_file" ]]; then
        cat "$name_file" | tr -d '[:space:]'
        return
    fi
    if [[ -n "${MODEL:-}" ]]; then
        echo "$MODEL"
        return
    fi
    echo "yolo11n"
}
MODEL="${MODEL:-$(detect_model)}"

# Автоопределение задачи по суффиксу модели
detect_task() {
    case "$1" in
        *-obb)       echo "obb" ;;
        *-seg)       echo "segment" ;;
        *-cls)       echo "classify" ;;
        *-pose)      echo "pose" ;;
        *-sem)       echo "semantic" ;;
        *)           echo "detect" ;;
    esac
}
TASK="${TASK:-$(detect_task "$MODEL")}"

# Автоопределение данных по задаче
detect_data_dir() {
    case "$1" in
        obb)         echo "/app/dotav1/images/val" ;;
        segment)     echo "/app/coco_val2017/images/val2017" ;;
        classify)    echo "/app/coco_val2017/images/val2017" ;;
        pose)        echo "/app/coco_val2017/images/val2017" ;;
        semantic)    echo "/app/cityscapes/images/val" ;;
        *)           echo "/app/coco_val2017/images/val2017" ;;
    esac
}
DATA_DIR="${DATA_DIR:-$(detect_data_dir "$TASK")}"

FPS_CMD="python3 /app/benchmark/fps_bench.py $MODEL $TASK --temp-dir /app/model --data-dir $DATA_DIR --warmup 10 --measure 20 --quantize none"

# Допустимое падение FPS (в %) — ниже этого считаем нестабильность
FPS_DROP_THRESHOLD=5

# Шаги частот для тестирования
CPU_FREQS=(1800000 2016000 2208000 2256000 2304000 2352000 2400000)
NPU_FREQS=(600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)
GPU_FREQS=(400000000 500000000 600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)

# ─── Утилиты ────────────────────────────────────────────────────────────────

require_root() {
    [[ $EUID -ne 0 ]] && { echo "[ОШИБКА] Запустите через: sudo $0 $*"; exit 1; }
}

die() { echo "[ОШИБКА] $*"; exit 1; }

require_docker() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || die "Контейнер $CONTAINER не запущен"
}

is_oc_installed() {
    cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "2352000"
}

freq_to_mhz() {
    local f=$1
    if [[ $f -ge 1000000000 ]]; then
        echo $((f / 1000000))
    elif [[ $f -ge 1000000 ]]; then
        echo $((f / 1000))
    else
        echo "$f"
    fi
}

get_max_temp() {
    local max=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local t
        t=$(cat "$zone" 2>/dev/null || echo 0)
        [[ $t -gt $max ]] && max=$t
    done
    echo $((max / 1000))
}

# Замер FPS — возвращает только число
measure_fps() {
    local result
    result=$(docker exec "$CONTAINER" $FPS_CMD 2>/dev/null | grep '^\{')
    if [[ -z "$result" ]]; then
        echo "0"
        return
    fi
    echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('npu_none',{}).get('fps',0))" 2>/dev/null || echo "0"
}

# Проверка стабильности: FPS не упал больше чем на порог
# $1 = baseline FPS, $2 = текущий FPS
# Возвращает 0 = стабильно, 1 = нестабильно
check_stability() {
    local baseline=$1
    local current=$2
    python3 -c "
b=$baseline; c=$current
if b <= 0: exit(1)
drop = (b - c) / b * 100
exit(0 if drop < $FPS_DROP_THRESHOLD else 1)
" 2>/dev/null
}

save_state() {
    local cpu=$1 npu=$2 gpu=$3
    mkdir -p "$OC_DIR"
    cat > "$OC_STATE" << EOF
# Сохранённая конфигурация разгона
# Дата: $(date '+%Y-%m-%d %H:%M:%S')
cpu_freq=$cpu
npu_freq=$npu
gpu_freq=$gpu
EOF
}

load_state() {
    if [[ -f "$OC_STATE" ]]; then
        source "$OC_STATE"
        echo "cpu=${cpu_freq:-0} npu=${npu_freq:-0} gpu=${gpu_freq:-0}"
    else
        echo "cpu=0 npu=0 gpu=0"
    fi
}

# ─── Управление частотами ───────────────────────────────────────────────────

set_cpu_freq() {
    local freq=$1
    for p in /sys/devices/system/cpu/cpufreq/policy{4,6}; do
        echo userspace > "$p/scaling_governor" 2>/dev/null
        echo "$freq" > "$p/scaling_setspeed" 2>/dev/null
    done
}

set_npu_freq() {
    local freq=$1
    echo userspace > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/min_freq 2>/dev/null
    echo "$freq" > /sys/class/devfreq/fdab0000.npu/max_freq 2>/dev/null
}

set_gpu_freq() {
    local freq=$1
    echo userspace > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/min_freq 2>/dev/null
    echo "$freq" > /sys/class/devfreq/fb000000.gpu/max_freq 2>/dev/null
}

restore_governors() {
    for p in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do
        echo ondemand > "$p/scaling_governor" 2>/dev/null
    done
    echo rknpu_ondemand > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null
    echo simple_ondemand > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null
}

# ─── Команды ─────────────────────────────────────────────────────────────────

cmd_status() {
    echo "════════════════════════════════════════════════════════"
    echo "  СОСТОЯНИЕ РАЗГОНА"
    echo "════════════════════════════════════════════════════════"
    is_oc_installed && echo "  DTB:   OC (2352/2400 МГц доступны)" || echo "  DTB:   стоковый"
    echo "  CPU6:  $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq) kHz ($(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor))"
    echo "  NPU:   $(cat /sys/class/devfreq/fdab0000.npu/cur_freq) Hz ($(cat /sys/class/devfreq/fdab0000.npu/governor))"
    echo "  GPU:   $(cat /sys/class/devfreq/fb000000.gpu/cur_freq) Hz ($(cat /sys/class/devfreq/fb000000.gpu/governor))"
    echo "  Темп:  $(get_max_temp)°C"
    echo ""
    if [[ -f "$OC_STATE" ]]; then
        echo "  Сохранённая конфигурация:"
        grep -v '^#' "$OC_STATE" | sed 's/^/    /'
    else
        echo "  Сохранённая конфигурация: отсутствует"
    fi
    echo "════════════════════════════════════════════════════════"
}

cmd_test() {
    local quick=0
    [[ "${1:-}" == "--quick" ]] && quick=1

    require_docker

    echo "════════════════════════════════════════════════════════"
    echo "  ТЕСТ СТАБИЛЬНОСТИ РАЗГОНА"
    echo "════════════════════════════════════════════════════════"
    echo "  Порог падения FPS: ${FPS_DROP_THRESHOLD}%"
    echo "════════════════════════════════════════════════════════"

    # Базовый FPS (стоковые частоты)
    restore_governors
    sleep 2
    echo ""
    echo "── Базовый FPS (сток) ──"
    local baseline
    baseline=$(measure_fps)
    echo "  NPU FPS: $baseline"
    [[ "$baseline" == "0" ]] && die "Не удалось замерить FPS"

    local best_cpu=$({ cat /sys/devices/system/cpu/cpufreq/policy6/cpuinfo_max_freq; } 2>/dev/null)
    local best_npu=1000000000
    local best_gpu=$({ cat /sys/class/devfreq/fb000000.gpu/available_frequencies | tr ' ' '\n' | sort -n | tail -1; } 2>/dev/null)

    # ─── Тест CPU ─────────────────────────────────────────────────────────
    echo ""
    echo "── Тест CPU ──"
    for freq in "${CPU_FREQS[@]}"; do
        local mhz
        mhz=$(freq_to_mhz "$freq")

        # Пропустить если частота недоступна
        cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "$freq" || continue

        set_cpu_freq "$freq"
        sleep 1
        local fps
        fps=$(measure_fps)
        local temp
        temp=$(get_max_temp)

        if check_stability "$baseline" "$fps"; then
            local gain
            gain=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✓\n" "$mhz" "$fps" "$gain" "$temp"
            best_cpu=$freq
        else
            local drop
            drop=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✗ НЕСТАБИЛЬНО\n" "$mhz" "$fps" "$drop" "$temp"
            [[ $quick -eq 1 ]] && break
        fi
    done

    # ─── Тест NPU ─────────────────────────────────────────────────────────
    echo ""
    echo "── Тест NPU ──"
    restore_governors
    sleep 1
    for freq in "${NPU_FREQS[@]}"; do
        local mhz
        mhz=$(freq_to_mhz "$freq")

        set_npu_freq "$freq"
        sleep 1
        local fps
        fps=$(measure_fps)
        local temp
        temp=$(get_max_temp)

        if check_stability "$baseline" "$fps"; then
            local gain
            gain=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✓\n" "$mhz" "$fps" "$gain" "$temp"
            best_npu=$freq
        else
            local drop
            drop=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✗ НЕСТАБИЛЬНО\n" "$mhz" "$fps" "$drop" "$temp"
            [[ $quick -eq 1 ]] && break
        fi
    done

    # ─── Тест GPU ─────────────────────────────────────────────────────────
    echo ""
    echo "── Тест GPU ──"
    restore_governors
    sleep 1
    for freq in "${GPU_FREQS[@]}"; do
        local mhz
        mhz=$(freq_to_mhz "$freq")

        set_gpu_freq "$freq"
        sleep 1
        local fps
        fps=$(measure_fps)
        local temp
        temp=$(get_max_temp)

        if check_stability "$baseline" "$fps"; then
            local gain
            gain=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✓\n" "$mhz" "$fps" "$gain" "$temp"
            best_gpu=$freq
        else
            local drop
            drop=$(python3 -c "print(f'{($fps - $baseline) / $baseline * 100:+.1f}%')" 2>/dev/null || echo "?")
            printf "  %5s МГц: %6.2f FPS (%s) %3d°C  ✗ НЕСТАБИЛЬНО\n" "$mhz" "$fps" "$drop" "$temp"
            [[ $quick -eq 1 ]] && break
        fi
    done

    # ─── Результат ────────────────────────────────────────────────────────
    restore_governors
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  МАКС. СТАБИЛЬНЫЕ ЧАСТОТЫ"
    echo "════════════════════════════════════════════════════════"
    echo "  CPU:  $(freq_to_mhz "$best_cpu") МГц"
    echo "  NPU:  $(freq_to_mhz "$best_npu") МГц"
    echo "  GPU:  $(freq_to_mhz "$best_gpu") МГц"
    echo "════════════════════════════════════════════════════════"

    save_state "$best_cpu" "$best_npu" "$best_gpu"
    echo "  Сохранено: $OC_STATE"
}

cmd_apply() {
    [[ -f "$OC_STATE" ]] || die "Нет сохранённой конфигурации. Запустите: $0 test"
    source "$OC_STATE"

    echo "════════════════════════════════════════════════════════"
    echo "  ПРИМЕНЕНИЕ РАЗГОНА"
    echo "════════════════════════════════════════════════════════"
    echo "  CPU: $(freq_to_mhz "${cpu_freq:-0}") МГц"
    echo "  NPU: $(freq_to_mhz "${npu_freq:-0}") МГц"
    echo "  GPU: $(freq_to_mhz "${gpu_freq:-0}") МГц"
    echo "════════════════════════════════════════════════════════"

    [[ "${cpu_freq:-0}" -gt 0 ]] && set_cpu_freq "$cpu_freq"
    [[ "${npu_freq:-0}" -gt 0 ]] && set_npu_freq "$npu_freq"
    [[ "${gpu_freq:-0}" -gt 0 ]] && set_gpu_freq "$gpu_freq"

    echo "[OK] Частоты установлены"
}

cmd_revert() {
    echo "════════════════════════════════════════════════════════"
    echo "  ОТКАТ К СТОКОВЫМ ЧАСТОТАМ"
    echo "════════════════════════════════════════════════════════"
    restore_governors
    echo "[OK] Governors восстановлены (ondemand)"
    echo "  CPU6: $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)"
    echo "  NPU:  $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
    echo "  GPU:  $(cat /sys/class/devfreq/fb000000.gpu/cur_freq)"
}

cmd_set() {
    require_docker

    for arg in "$@"; do
        case "$arg" in
            cpu=*) set_cpu_freq "${arg#cpu=}" ;;
            npu=*) set_npu_freq "${arg#npu=}" ;;
            gpu=*) set_gpu_freq "${arg#gpu=}" ;;
        esac
    done

    sleep 1
    local fps
    fps=$(measure_fps)
    local temp
    temp=$(get_max_temp)
    echo "Установлено. FPS: $fps, Темп: ${temp}°C"
}

cmd_monitor() {
    require_docker
    echo "════════════════════════════════════════════════════════"
    echo "  МОНИТОРИНГ СТАБИЛЬНОСТИ"
    echo "  Проверка FPS каждые 60 сек, автооткат при падении"
    echo "  Ctrl+C для остановки"
    echo "════════════════════════════════════════════════════════"

    # Замерить baseline
    local baseline
    baseline=$(measure_fps)
    echo "  Baseline FPS: $baseline"
    echo ""

    local check_count=0
    local fail_count=0

    while true; do
        sleep 60
        check_count=$((check_count + 1))
        local fps
        fps=$(measure_fps)
        local temp
        temp=$(get_max_temp)
        local ts
        ts=$(date +%H:%M:%S)

        if check_stability "$baseline" "$fps"; then
            printf "  [%s] #%d  FPS: %6.2f  Темп: %3d°C  ✓\n" "$ts" "$check_count" "$fps" "$temp"
            fail_count=0
        else
            fail_count=$((fail_count + 1))
            printf "  [%s] #%d  FPS: %6.2f  Темп: %3d°C  ✗ ПАДЕНИЕ (x%d)\n" "$ts" "$check_count" "$fps" "$temp" "$fail_count"

            if [[ $fail_count -ge 3 ]]; then
                echo ""
                echo "  ══ 3 подряд падения FPS — ОТКАТ ══"
                restore_governors
                echo "  Governors восстановлены."
                fail_count=0
                # Перезамерить baseline
                sleep 5
                baseline=$(measure_fps)
                echo "  Новый baseline: $baseline"
            fi
        fi
    done
}

# ─── Точка входа ─────────────────────────────────────────────────────────────

case "${1:-help}" in
    status)  cmd_status ;;
    test)    require_root "$@"; cmd_test "${2:-}" ;;
    apply)   require_root "$@"; cmd_apply ;;
    revert)  require_root "$@"; cmd_revert ;;
    set)     require_root "$@"; shift; cmd_set "$@" ;;
    monitor) require_root "$@"; cmd_monitor ;;
    *)
        echo "Использование: $(basename "$0") <команда>"
        echo ""
        echo "Команды:"
        echo "  status              Текущее состояние"
        echo "  test [--quick]      Тест всех частот + поиск макс. стабильной"
        echo "  apply               Применить сохранённую конфигурацию"
        echo "  revert              Откат к стоковым частотам"
        echo "  set cpu=X npu=Y gpu=Z  Установить конкретные частоты"
        echo "  monitor             Мониторинг + автооткат при нестабильности"
        ;;
esac
