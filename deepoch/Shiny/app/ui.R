library(shiny)
library(DT)

shinyUI(pageWithSidebar(
  headerPanel(''),
  sidebarPanel(
    fileInput(inputId = 'file',
              label = 'Select Image File:'),
    actionButton("apply", 'Apply')
  ),
  mainPanel(
    h3('Result'),
    imageOutput('image'),
    DT::dataTableOutput('table')
  )
))
