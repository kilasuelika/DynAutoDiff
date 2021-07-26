import delimited churn_data.csv, varn(1) clear
sa churn.dta, replace
import delimited customer_data.csv,  varn(1) clear
sa customer, replace
import delimited internet_data.csv,  varn(1) clear
sa internet, replace

use churn, clear
merge 1:1 customerid using customer, nogen
merge 1:1 customerid using internet, nogen

drop customerid
foreach x of varlist * {
   capture confirm string variable `x'
   if !_rc {
      encode `x', gen(`x'_g)
	  drop `x'
	  rename `x'_g `x'
   }
   else {
      
   }
}

ds //列出所有变量进宏r(varlist)
local allvars `r(varlist)'
local yvar churn
local xvars: list allvars - yvar
di "`xvars'"
replace churn=churn-1

logistic `yvar' `xvars', coef log trace showstep


preserve
drop churn
export delimited "../../Xb.txt", novarn nolab delim(" ") replace

restore
keep churn
replace churn=churn-1
export delimited "../../yb.txt", novarn nolab  delim(" ") replace

