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
library(pubmedR)
rm(list=ls())



# List of major Swiss terms and institutions for filtering
swiss_terms <- c(
  "Swiss", "Switzerland", "Schweiz", "Suisse", "Svizzera",
  "University of Zurich", "ETH Zurich", "EPFL", 
  "University of Geneva", "University of Basel",
  "University of Bern", "University of Lausanne",
  "University of St. Gallen", "University of Fribourg",
  "Universität Zürich", "Université de Genève",
  "Universität Basel", "Universität Bern",
  "Université de Lausanne", "Universität St. Gallen",
  "Université de Fribourg","Università della Svizzera italiana","University of Lugano"
)


# Set up query parameters
query <- "organoid[Title/Abstract] AND (2018[Publication Date]:2025[Publication Date]) AND Switzerland[Affiliation]"
query <- "organoid[Title/Abstract] AND (2000[Publication Date]:2025[Publication Date])"


query <- "'Alzheimer Disease'[MeSH:noexp] AND (2010[Publication Date] : 2015[Publication Date] ) AND Switzerland[Affiliation]"
query <- "'Alzheimer Disease'[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"

query <- "'Multiple Sclerosis'[MeSH:noexp] AND (2010[Publication Date] : 2015[Publication Date] ) AND Switzerland[Affiliation]"
query <- "'Multiple Sclerosis'[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"

query <- "Neoplasms[MeSH:noexp] AND (2010[Publication Date] : 2015[Publication Date] ) AND Switzerland[Affiliation]"
query <- "Neoplasms[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"

query <- "Stroke[MeSH:noexp] AND (2010[Publication Date] : 2015[Publication Date] ) AND Switzerland[Affiliation]"
query <- "Stroke[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"


api_key <- "3e154bf167c356cce640937945315c7ff408"

# Run the query
res <- pmQueryTotalCount(query = query, api_key = api_key)
cat("Total research publications:", res$total_count, "\n")

# Fetch the data
papers <- pmApiRequest(query = query, limit = min(res$total_count, 3000), 
                       api_key = api_key)

save(papers,file = "output/save_api_temp.Rdata")
load("output/save_api_temp.Rdata")

# Convert to data frame
df <- pmApi2df(papers)
df <- df[, c("AU","AF","TI","SO","DT","AU_UN","MESH","AB","J9","PY")]



df$split_AU <- strsplit(as.character(df$AU), split = ";\\s*")
df$count_AU <- unlist(lapply(df$split_AU_UN,length))

df$split_AU_UN <- strsplit(as.character(df$AU_UN), split = ";\\s*")
df$count_affil <- unlist(lapply(df$split_AU_UN,length))

df<- df[df$AU_UN != "",]

## checking that the authors and affiliations match ####
df_temp <- df
df_temp$count_diff <- 0
df_temp$count_diff[df$count_AU != df$count_affil] <- 1
if(sum(df_temp$count_diff) == 0){
  print("Same count for authors and affiliations")
} else {
  print("!!Different count for authors and affiliations!!")
}
rm(df_temp)

## making a dummy 0/1 for the swiss affiliation matches for each coauthor ####
contains_swiss_term_1 <- function(texts) {
  lapply(texts, function(text) {
    if (any(sapply(swiss_terms, function(term) grepl(term, text, ignore.case = TRUE)))) {
      return(1)
    }
    return(0)
  })
}

contains_swiss_term <- function(temp_ob) {
  return(unlist(lapply(temp_ob, contains_swiss_term_1)))
}
# 
# 
# ## Claude alternative 1 ####
# contains_swiss_term <- function(temp_ob) {
#   result <- lapply(temp_ob, function(text) {
#     if (any(sapply(swiss_terms, function(term) grepl(term, text, ignore.case = TRUE)))) {
#       return(1)
#     }
#     return(0)
#   })
# 
#   return(unlist(result))
# }
# 
# ## Claude alternative 2 ####
# contains_swiss_term <- function(temp_ob) {
#   sapply(temp_ob, function(text) {
#     as.integer(any(sapply(swiss_terms, function(term) grepl(term, text, ignore.case = TRUE))))
#   })
# }

df$swiss_affil <- lapply(df$split_AU_UN,contains_swiss_term)
df$swiss_affil_count <- as.integer(lapply(df$swiss_affil,sum))

df$count_AU <- unlist(lapply(df$split_AU_UN,length))

df$swiss_affil_last_author <- NA
for(l in 1:nrow(df)){
  temp_list <- unlist(df$swiss_affil[l])
  df$swiss_affil_last_author[l]<- temp_list[unlist(df$count_AU[l])]
}

df$swiss_affil_first_author <- NA
for(l in 1:nrow(df)){
  temp_list <- df$swiss_affil[l]
  df$swiss_affil_first_author[l]<- temp_list[[1]]
}


sum(df$swiss_affil_count)
sum(unlist(df$swiss_affil_last_author))
sum(unlist(df$swiss_affil_first_author))
max(df$swiss_affil_count)

df$count <- 1
counts_check_year <- df[df$swiss_affil_count != 0 ,] %>% group_by(PY) %>% summarize(count = sum(count))
counts_check_year_first_AU <- df[df$swiss_affil_first_author != 0 ,] %>% group_by(PY) %>% summarize(count = sum(count))
counts_check_year_last_AU <- df[df$swiss_affil_last_author != 0 ,] %>% group_by(PY) %>% summarize(count = sum(count))

counts_check_year<- left_join(counts_check_year,counts_check_year_first_AU, by = c("PY" = "PY"))
counts_check_year <-left_join(counts_check_year,counts_check_year_last_AU, by = c("PY" = "PY"))
names(counts_check_year) <- c("Year","Any Swiss","First Author","Last author")
print(counts_check_year)
sum(counts_check_year$`Last author`[4:8])
sum(counts_check_year$`Any Swiss`[4:8])
sum(counts_check_year$`First Author`[4:8])
