#!/usr/bin/env bash
# Runs the same Locust flood against 1, 2 and 4 API containers and saves CSVs in results/.
# Needs: docker compose, and `pip install -r requirements-dev.txt`
set -e
USERS=${USERS:-50}
SPAWN=${SPAWN:-5}
DURATION=${DURATION:-90s}
mkdir -p results
for N in 1 2 4; do
  echo "=== $N container(s) ==="
  docker compose up -d --build --scale api=$N
  docker compose restart nginx            # nginx must re-resolve the api containers
  echo "waiting for the model to load..."
  until curl -sf http://localhost:8080/api/health > /dev/null; do sleep 3; done
  sleep 20
  locust -f locust/locustfile.py --host http://localhost:8080 --headless \
         -u $USERS -r $SPAWN -t $DURATION --csv results/containers_$N --only-summary
done
docker compose down
echo "Done. See results/containers_*_stats.csv"
