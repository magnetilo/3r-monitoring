

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
mesh_depth <- 5
df_mesh <- df_mesh[df_mesh$tree_step < mesh_depth &
                     ( df_mesh$neoplasm_dumm == 1 |
                         df_mesh$nervous_dumm == 1),]


years_list <- c(2000:2025)

#query <- "Adenocarcinoma[MeSH:noexp] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"
#query <- "C04.182.117[MeSH] AND (2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation]"
#query <- "(2020[Publication Date] : 2025[Publication Date] ) AND Switzerland[Affiliation])"
#query <- paste0(tree_number, "[MeSH Tree]")

pub_counts <- data.frame(
  "term" = character(0),
  "year" = integer(0),
  "swiss" = integer(0),
  "total" = integer(0),
  stringsAsFactors = FALSE
)



for(t in 1:nrow(df_mesh)){
  for(y in 1:length(years_list)){
    query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND ",years_list[y],"[Publication Date] AND Switzerland[Affiliation]")
    res_ch <- pmQueryTotalCount(query = query, api_key = api_key)
  
    query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND ",years_list[y],"[Publication Date]")
    res_all <- pmQueryTotalCount(query = query, api_key = api_key)
    
    temp_row <- data.frame("term" = df_mesh$DescriptorName[t], "year" = years_list[y], "swiss" = res_ch$total_count, "total" = res_all$total_count)
    
    pub_counts <- rbind(pub_counts, temp_row)
  }
}


for(t in 194:nrow(df_mesh)){
  for(y in 1:length(years_list)){
    query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND ",years_list[y],"[Publication Date] AND Switzerland[Affiliation]")
    res_ch <- pmQueryTotalCount(query = query, api_key = api_key)
    
    query <- paste0(df_mesh$DescriptorName[t],"[MeSH:noexp] AND ",years_list[y],"[Publication Date]")
    res_all <- pmQueryTotalCount(query = query, api_key = api_key)
    
    temp_row <- data.frame("term" = df_mesh$DescriptorName[t], "year" = years_list[y], "swiss" = res_ch$total_count, "total" = res_all$total_count)
    
    pub_counts <- rbind(pub_counts, temp_row)
  }
}


pub_counts$swiss_share <- pub_counts$swiss/pub_counts$total


save(pub_counts, file = paste0("output/publication_counts_meshterm_swiss_all_mesh_",mesh_depth,".Rdata"))
write.csv(pub_counts, file = paste0("output/publication_counts_meshterm_swiss_all_",mesh_depth,".csv"))


df <- left_join(pub_counts,df_mesh, by = c("term" = "DescriptorName"))


df <- df[df$tree_step ==4,]



# Calculate 5-year running average
pub_counts_smooth <- df %>%
  arrange(term, year) %>%
  group_by(term) %>%
  mutate(
    swiss_share_smooth = zoo::rollmean(swiss_share, k = 5, fill = NA, align = "center")
  ) %>%
  ungroup()

# Find the 5 terms with greatest increase over the period
top_increase_terms <- pub_counts_smooth %>%
  group_by(term) %>%
  summarise(
    first_year = min(year[!is.na(swiss_share_smooth)]),
    last_year = max(year[!is.na(swiss_share_smooth)]),
    first_value = swiss_share_smooth[year == first_year][1],
    last_value = swiss_share_smooth[year == last_year][1],
    increase = last_value - first_value,
    .groups = "drop"
  ) %>%
  filter(!is.na(increase)) %>%
  arrange(desc(increase)) %>%
  slice_head(n = 5) %>%
  pull(term)

# Add a column to identify top increase terms
pub_counts_smooth <- pub_counts_smooth %>%
  mutate(
    is_top_increase = term %in% top_increase_terms,
    line_type = ifelse(is_top_increase, "Top 5 Increase", "Other Terms")
  )

# Create the plot
p <- ggplot(data = pub_counts_smooth, aes(x = year, y = swiss_share_smooth, color = term)) +
  geom_line(aes(group = term, linetype = line_type), size = 1) +
  geom_point(aes(text = paste("Term:", term,
                              "<br>Year:", year,
                              "<br>Swiss Publications:", swiss,
                              "<br>Total Publications:", total,
                              "<br>Swiss Share:", scales::percent(swiss_share_smooth, accuracy = 0.1))),
             size = 2) +
  scale_linetype_manual(values = c("Top 5 Increase" = "solid", "Other Terms" = "dashed")) +
  labs(
    title = "Swiss Share Over Years by Term (5-Year Running Average)",
    subtitle = "Dashed lines show the 5 terms with greatest increase over the period",
    x = "Year",
    y = "Swiss Share (5-Year Running Average)",
    color = "Term",
    linetype = "Line Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  scale_y_continuous(labels = scales::percent_format())

# Convert to interactive plotly chart
ggplotly(p, tooltip = "text")






