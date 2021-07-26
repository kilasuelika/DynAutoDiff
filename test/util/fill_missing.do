import delimited cancer_reg.txt, clear  delimiter(" ")

foreach var of varlist v1-v28 {
	replace `var'=0 if `var'==.
}

export delimited cancer_reg.txt, replace delimiter(" ") novarn