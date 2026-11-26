FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/nginx.conf
# the pages sit at the root; their CSS and JavaScript live under /static/, as they do in the FastAPI app
COPY tomato_leaf_checking_app/static/*.html /usr/share/nginx/html/
COPY tomato_leaf_checking_app/static/ /usr/share/nginx/html/static/
