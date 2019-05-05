library(httr)
library(shiny)

shinyServer(function(input, output) {
  api_host <- 'http://10.41.9.210:5000/predict'
  passData <- eventReactive(input$apply, {
    filepath <- input$file$datapath
    contentType <- input$file$type
    r <- POST(url = api_host, body = list(image = upload_file(filepath)))
    dfm <- as.data.frame(matrix(unlist(content(r, "parsed", "application/json")$predictions), 
                                ncol = 2, byrow = TRUE))
    rownames(dfm) <- 1:nrow(dfm)
    colnames(dfm) <- c('Class', 'Score')
    list(dfm=dfm, path=filepath, type=contentType)
  })

  output$table <- DT::renderDataTable(
    datatable(data = passData()$dfm)
  )

  output$image <- renderImage({
    list(src = passData()$path, contentType=passData()$type, width=600)
  }, deleteFile = TRUE)
})

