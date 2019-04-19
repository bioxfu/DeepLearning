sudo docker kill shiny

sudo docker rm shiny

sudo docker run --name shiny -d -p 8686:5050 -v $PWD/app:/srv/shiny-server/ bioxfu/shiny-server

sudo docker exec -ti shiny /bin/bash

# in docker container
# cd /srv/shiny-server/
# R
# > library(shiny)
# > runApp(host='0.0.0.0', port=5050)
