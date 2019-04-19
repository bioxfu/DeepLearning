sudo docker kill shiny

sudo docker rm shiny

sudo docker run --name shiny -d -p 8686:3838 -v $PWD/app:/srv/shiny-server/ bioxfu/shiny-server
