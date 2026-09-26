.PHONY: bench bench-pilot bench-model report

# Полный прогон всех моделей из benchmark/results (convert+validate+bench+eval)
# + автогенерация отчёта в benchmark/results/report/ (make_report.py).
# Resume: готовые модели пропускаются (benchmark/results/results.json).
# Фон: tmux new-session -d -s bench 'make bench > /tmp/bench.log 2>&1'
bench:
	python3 benchmark/benchmark.py

# Только PILOT-модели (минимальные размеры)
bench-pilot:
	python3 benchmark/benchmark.py --pilot

# Одна модель: make bench-model M=yolo26n
bench-model:
	python3 benchmark/benchmark.py --model $(M)

# Пересобрать отчёт из готового журнала результатов без перегона моделей.
# Запускает make_report.py в convert-контейнере (на хосте нет matplotlib),
# результаты монтируются writable, как в run_report().
report:
	docker compose -f convert/docker-compose.yml run --rm \
		-v "$(CURDIR)/benchmark/results:/app/benchmark/results" \
		--entrypoint python convert \
		/app/benchmark/make_report.py --out /app/benchmark/results/report