

# Install and load required packages
library(pubmedR) ## also useful library: library(easyPubMed)
library(ggplot2)
library(plyr)
library(dplyr)
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
rm(list=ls())

api_key <- "3e154bf167c356cce640937945315c7ff408"

load("output/mesh_terms.Rdata")


# reduce the size of MeSH term selection 
df_mesh <- df_mesh[df_mesh$tree_step < 10 &
                     ( df_mesh$neoplasm_dumm == 1 |
                     df_mesh$nervous_dumm == 1),]


#query <- "Adenocarcinoma[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"
#query <- "C04.182.117[MeSH] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"
#query <- "(2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation])"
#query <- paste0(tree_number, "[MeSH Tree]")


pub_counts <- data.frame("term" = df_mesh$DescriptorName,
                         "count_t1" = 0,
                         "count_t2" = 0)

for(t in 1:nrow(df_mesh)){
  query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND (2010[Publication Date] : 2015[Publication Date] ) AND Switzerland[Affiliation]")
  res <- pmQueryTotalCount(query = query, api_key = api_key)
  pub_counts$count_t1[t] <- res$total_count
  
  query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]")
  res <- pmQueryTotalCount(query = query, api_key = api_key)
  pub_counts$count_t2[t] <- res$total_count
}


save(pub_counts, file = "output/publication_counts_meshterm.Rdata")
write.csv(pub_counts, file = "output/publication_counts_meshterm.csv")





