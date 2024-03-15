
clear all

global path "fixed_effects\"
global result "fixed_effects\"

****************************************pop*************************************

use "$path/panel_U.dta",clear



gen xtime_in=time-interv_time+3888000/9*2 if (interv_time~=0)
gen difference=floor(xtime_in/7776000*9/2)
tab difference
*keep if value_c==1
*tab difference
replace difference=-99 if(difference==.)

gen is_in=0
replace is_in=1 if(difference>=0)

gen is_in2=0
replace is_in2=1 if(time>=interv_time)

gen dd=time-interv_time

gen is_in3=0
replace is_in3=1 if(dd>=0)






*xtset id time
// drop if region!=1
// bysort city: gen ct_sum = sum(saps)
// drop if city == 810000

gen treated = is_interv
// gen post_ = difference
// gen inde = value

// gen did = post_ * treated

// encode city, g(city_id)
// drop period_differenve==0
// forvalues i = 3(-1)1{
// gen pre_`i' = (period_difference == -`i' & treated == 1)
// }
// // gen current = (period_difference == 0 & treated == 1)
// forvalues j = 0(1)3{
// gen  time_`j' = (period_difference == `j' & treated == 1)
//  }
// drop time_0

// reghdfe inde  pre_* current  time_*  , absorb(id time) cluster(id)
// reghdfe value_c  pre_*  time_*  , absorb(id time_period#region) cluster(id)
// reghdfe value_c  pre_*  time_*  , absorb(id time_period) cluster(id)
reghdfe value_c is_in3 , absorb(id time#region) cluster(id) 
// reghdfe value_c is_in2 , absorb(id time) cluster(id) 
// ssc install colrspace
//
// colorpalette HTML, globals
egen mean=mean(value_c)