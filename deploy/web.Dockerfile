FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/nginx.conf
# the pages sit at the root; their CSS and JavaScript live under /static/, as they do in the FastAPI app
COPY app/static/*.html /usr/share/nginx/html/
COPY app/static/ /usr/share/nginx/html/static/
