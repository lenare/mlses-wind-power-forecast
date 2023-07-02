# General Notes

## Feature Selection

Things to consider:

1. Wind Speed: Wind speed is a crucial factor in determining wind power generation. It is typically measured at the turbine hub height and is available from weather stations or numerical weather prediction models.

2. Wind Direction: Wind direction provides information about the prevailing wind patterns and can impact turbine performance. It is often measured in degrees or represented as categorical variables (e.g., North, South, East, West).

3. Temperature: Ambient temperature can affect air density, which influences wind turbine performance. Higher temperatures can lead to reduced power output due to lower air density.

4. Air Pressure: Air pressure influences wind behavior and can impact wind power generation. Changes in air pressure can be associated with weather fronts, which affect wind patterns.

5. Humidity: Humidity affects air density and can have an indirect impact on wind power output. Lower humidity levels may result in lower air density, potentially affecting turbine performance.

6. Precipitation: Precipitation, such as rain or snow, can affect turbine operation and maintenance. Extreme weather conditions may lead to reduced power output or even temporary shutdowns.

7. Seasonality: Seasonal patterns play a significant role in wind power generation. Different seasons can exhibit variations in wind patterns, which can impact power output. Including seasonal indicators or variables can help capture these patterns.

8. Time of Day: The time of day can influence wind patterns and, consequently, wind power generation. Diurnal variations in wind speed and direction should be considered.

9. Turbine Characteristics: Some turbine-specific variables can provide additional information for better predictions. These may include turbine capacity, rotor diameter, hub height, or other technical specifications.

etc.

## Next steps

- plot time series to see if there are patterns you can exploit
- Look at correlations (e.g. Pearson correlation) between variables in data set

## Notes

- Density adjusted wind speed is equal to or correlates with wind speed or other wind speed measurements?
