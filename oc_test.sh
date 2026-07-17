#!/usr/bin/env bash
# Тест разгона: FPS до → DTB OC → FPS после.
# Автоматически определяет нестабильность по падению FPS.
#
# Использование:
#   ./oc_test.sh              # полный цикл (FPS до → DTB OC → ребут → FPS после)
#   ./oc_test.sh --fps-only   # только FPS с текущим DTB
#   ./oc_test.sh --oc-setup   # только DTB модификация + ребут
#   ./oc_test.sh --compare    # FPS на всех частотах (до и после)
set -uo pipefail
cd "$(dirname "$0")"

CONTAINER="percept"
TASK="detect"
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

FPS_CMD="python3 /app/benchmark/fps_bench.py $MODEL $TASK --temp-dir /app/model --data-dir /app/coco_val2017/images/val2017 --warmup 10 --measure 20 --quantize none"

FPS_ONLY=0
OC_SETUP_ONLY=0
COMPARE=0

for arg in "$@"; do
    case "$arg" in
        --fps-only)  FPS_ONLY=1 ;;
        --oc-setup)  OC_SETUP_ONLY=1 ;;
        --compare)   COMPARE=1 ;;
    esac
done

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[ОШИБКА] Контейнер $CONTAINER не запущен."
    exit 1
fi

is_oc_installed() {
    cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "2352000"
}

measure_fps() {
    docker pause "$CONTAINER" 2>/dev/null || true
    local result
    result=$(docker exec "$CONTAINER" $FPS_CMD 2>/dev/null | grep '^\{')
    docker unpause "$CONTAINER" 2>/dev/null || true
    echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"npu_none\",{}).get(\"fps\",0):.2f}')" 2>/dev/null || echo "0"
}

freq_to_mhz() {
    local f=$1
    [[ $f -ge 1000000000 ]] && echo $((f / 1000000)) && return
    [[ $f -ge 1000000 ]] && echo $((f / 1000)) && return
    echo "$f"
}

# ─── Только FPS ──────────────────────────────────────────────────────────────

if [[ $FPS_ONLY -eq 1 ]]; then
    echo "════════════════════════════════════════════════════════"
    echo "  FPS ТЕСТ"
    echo "════════════════════════════════════════════════════════"
    echo "  CPU6: $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq) kHz"
    echo "  NPU:  $(cat /sys/class/devfreq/fdab0000.npu/cur_freq) Hz"
    echo "  GPU:  $(cat /sys/class/devfreq/fb000000.gpu/cur_freq) Hz"
    is_oc_installed && echo "  OC DTB: установлен" || echo "  OC DTB: стоковый"
    echo "════════════════════════════════════════════════════════"
    local_fps=$(measure_fps)
    echo "  NPU FPS: $local_fps"
    exit 0
fi

# ─── Только OC DTB ───────────────────────────────────────────────────────────

apply_oc_dtb() {
    if is_oc_installed; then
        echo "[OK] OC DTB уже установлен"
        return 0
    fi
    [[ -f "$OC_DTB_BAK" ]] || { echo "[ОШИБКА] Бэкап не найден"; exit 1; }

    echo "  Декомпиляция DTB..."
    dtc -I dtb -O dts -o "$OC_DTS" "$OC_DTB" 2>/dev/null
    [[ -f "$OC_DTS" ]] || { echo "[ОШИБКА] Декомпиляция failed"; exit 1; }

    echo "  CPU: расблокировка 2352/2400 МГц..."
    sed -i '/opp-2352000000/,/};/ { /opp-supported-hw/s/=.*/= <0xffffffff 0xffffffff>;/ }' "$OC_DTS"
    sed -i '/opp-2400000000/,/};/ { /opp-supported-hw/s/=.*/= <0xffffffff 0xffffffff>;/ }' "$OC_DTS"

    echo "  NPU/GPU: добавление 1050/1100 МГц..."
    local npu_opp='			opp-1050000000 {
				opp-supported-hw = <0xf9 0xffffffff>;
				opp-hz = <0x00 0x3e919e40>;
				opp-microvolt = <0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0 0xdbba0>;
			};
			opp-1100000000 {
				opp-supported-hw = <0xf9 0xffffffff>;
				opp-hz = <0x00 0x4190ce80>;
				opp-microvolt = <0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0 0xe7ef0>;
			};'
    # Вставить после каждого opp-1000000000 блока (GPU и NPU)
    local gpu_line npu_line
    gpu_line=$(grep -n "opp-1000000000 {" "$OC_DTS" | head -1 | cut -d: -f1)
    npu_line=$(grep -n "opp-1000000000 {" "$OC_DTS" | tail -1 | cut -d: -f1)
    # Найти закрытие каждого блока
    for anchor in $npu_line $gpu_line; do
        local end
        end=$(awk -v start="$anchor" 'NR>=start && /^\t\t\};/{print NR; exit}' "$OC_DTS")
        [[ -n "$end" ]] && sed -i "${end}r /dev/stdin" "$OC_DTS" <<< "$npu_opp"
    done

    echo "  Компиляция DTB..."
    dtc -I dts -O dtb -o /tmp/oc_new.dtb "$OC_DTS" 2>/dev/null
    [[ -s /tmp/oc_new.dtb ]] || { echo "[ОШИБКА] Компиляция failed"; exit 1; }

    echo 'orangepi' | sudo -S cp /tmp/oc_new.dtb "$OC_DTB"
    echo "[OK] OC DTB установлен"
}

if [[ $OC_SETUP_ONLY -eq 1 ]]; then
    echo "════════════════════════════════════════════════════════"
    echo "  УСТАНОВКА OC DTB"
    echo "════════════════════════════════════════════════════════"
    apply_oc_dtb
    echo "  Перезагрузка через 5 сек..."
    sleep 5
    echo 'orangepi' | sudo -S reboot
    exit 0
fi

# ─── Сравнение FPS на всех частотах ──────────────────────────────────────────

if [[ $COMPARE -eq 1 ]]; then
    echo "════════════════════════════════════════════════════════"
    echo "  СРАВНЕНИЕ FPS НА РАЗНЫХ ЧАСТОТАХ"
    echo "════════════════════════════════════════════════════════"
    echo "  Базовый FPS (все на ondemand)..."
    echo 'orangepi' | sudo -S bash -c '
        for p in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do echo ondemand > "$p/scaling_governor"; done
        echo rknpu_ondemand > /sys/class/devfreq/fdab0000.npu/governor
    ' 2>/dev/null
    sleep 2
    baseline=$(measure_fps)
    echo "  Baseline: $baseline FPS"
    echo ""

    echo "  CPU разгон (NPU на ondemand):"
    for freq in 2208000 2256000 2304000 2352000 2400000; do
        cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "$freq" || continue
        echo 'orangepi' | sudo -S bash -c "
            for p in /sys/devices/system/cpu/cpufreq/policy{4,6}; do
                echo userspace > \$p/scaling_governor; echo $freq > \$p/scaling_setspeed
            done
        " 2>/dev/null
        sleep 1
        fps=$(measure_fps)
        mhz=$(freq_to_mhz "$freq")
        drop=$(python3 -c "b=$baseline; c=$fps; print(f'{(c-b)/b*100:+.1f}%' if b>0 else '?')" 2>/dev/null)
        printf "    %5s МГц: %s FPS (%s)\n" "$mhz" "$fps" "$drop"
    done

    echo ""
    echo "  NPU разгон (CPU на ondemand):"
    echo 'orangepi' | sudo -S bash -c '
        for p in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do echo ondemand > "$p/scaling_governor"; done
    ' 2>/dev/null
    for freq in 600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000; do
        echo 'orangepi' | sudo -S bash -c "
            echo userspace > /sys/class/devfreq/fdab0000.npu/governor
            echo $freq > /sys/class/devfreq/fdab0000.npu/min_freq
            echo $freq > /sys/class/devfreq/fdab0000.npu/max_freq
        " 2>/dev/null
        sleep 1
        fps=$(measure_fps)
        mhz=$(freq_to_mhz "$freq")
        drop=$(python3 -c "b=$baseline; c=$fps; print(f'{(c-b)/b*100:+.1f}%' if b>0 else '?')" 2>/dev/null)
        printf "    %5s МГц: %s FPS (%s)\n" "$mhz" "$fps" "$drop"
    done

    echo 'orangepi' | sudo -S bash -c '
        echo rknpu_ondemand > /sys/class/devfreq/fdab0000.npu/governor
        for p in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do echo ondemand > "$p/scaling_governor"; done
    ' 2>/dev/null
    echo ""
    echo "════════════════════════════════════════════════════════"
    exit 0
fi

# ─── Полный цикл ─────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  ТЕСТ РАЗГОНА: FPS ДО → DTB ОС → ПЕРЕЗАГРУЗКА"
echo "════════════════════════════════════════════════════════"

echo ""
echo "── FPS ДО разгона (стоковый DTB) ──"
FPS_BEFORE=$(measure_fps)
echo "  NPU FPS ДО: $FPS_BEFORE"

echo ""
apply_oc_dtb

echo ""
echo "════════════════════════════════════════════════════════"
echo "  FPS ДО: $FPS_BEFORE"
echo "  OC DTB установлен. Перезагрузка..."
echo ""
echo "  После перезагрузки запустите:"
echo "    sudo ./oc_test.sh --fps-only"
echo "    sudo ./oc_test.sh --compare"
echo "════════════════════════════════════════════════════════"
sleep 5
echo 'orangepi' | sudo -S reboot
