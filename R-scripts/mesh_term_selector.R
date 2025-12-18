## mesh data import ##


library(plyr)
library(dplyr)# used in phenology.R, bees.R
library(ggplot2) # used in beehours_varying_temp.R   regr_apple.R
library(readxl)
library(tidyr)
library(lubridate)
library(lattice)
library(stringr)
library(htmlTable)
library(plotly)
library(manipulateWidget)
library(shiny)
library(ggiraph)
library(pubmedR)
rm(list=ls())

# Load required libraries
library(XML)

## https://en.wikipedia.org/wiki/List_of_MeSH_codes

xml_file <- "mesh_data/desc2025.xml"
# Parse the XML with validation options if DTD is referenced in XML
xml_data <- xmlParse(xml_file)

# Once parsed, extract data
root <- xmlRoot(xml_data)

# Create an empty list to store the extracted data
data_list <- list()

# Extract data based on structure
for (node in xmlChildren(root)) {
  # Process each record
  record <- xmlSApply(node, xmlValue)
  data_list[[length(data_list) + 1]] <- record
}

df_mesh <- data.frame("DescriptorName" = as.character(),
                     "TreeNumberList" = as.character())
for(t in 1:length(data_list)){
  temp_line <- c("term" = data_list[[t]]['DescriptorName'],
                 "mesh_num" = data_list[[t]]['TreeNumberList'])
  df_mesh<- rbind(df_mesh,temp_line)
  }
names(df_mesh) <- c("DescriptorName","TreeNumberList")

df_mesh$neoplasm_dumm <- NA
contains_C04 <- function(observation) {
  if (grepl("\\bC04", observation)) {
    return(1)
  } else {
    return(0)
  }
}

df_mesh$neoplasm_dumm <- sapply(df_mesh$TreeNumberList,contains_C04)
sum(df_mesh$neoplasm_dumm)

df_mesh$nervous_dumm <- NA
contains_C10 <- function(observation) {
  if (grepl("\\bC10", observation)) {
    return(1)
  } else {
    return(0)
  }
}
df_mesh$nervous_dumm <- sapply(df_mesh$TreeNumberList,contains_C10)
sum(df_mesh$nervous_dumm)


count_points <- function(input_string) {
  # Count the number of decimal points in the string
  point_count <- lengths(regmatches(input_string, gregexpr("\\.", input_string)))
  return(point_count)
}

df_mesh$tree_step <- NA
df_mesh$tree_step <- sapply(df_mesh$TreeNumberList,count_points)
hist(df_mesh$tree_step)


save(df_mesh, file = "output/mesh_terms.Rdata")
