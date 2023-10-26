package main

import (
	"fmt"
	"strings"
)

func main() {
	// costsInOneDay := [][]float32{{7, 1}, {1, 7}}
	costsInOneDay := [][]float32{{5}, {5}}
	dayNum := 21
	displayBeforeCal(costsInOneDay, dayNum)

	notCutPrice := getNoCutCost(costsInOneDay, dayNum)
	onlyDiscountCost := getOnlyDiscountCost(costsInOneDay, dayNum)
	beforeCutPrice := getBeforeCutCost(costsInOneDay, dayNum)
	newCutPrice := getNewCutCost(costsInOneDay, dayNum)

	displayAfterCal(notCutPrice, onlyDiscountCost, beforeCutPrice, newCutPrice)
}

func displayBeforeCal(costsInOneDay [][]float32, dayNum int) {
	fmt.Printf("####计算杭州每月通勤费用####\n")
	for i, v := range costsInOneDay {
		strs := []string{}
		for _, v1 := range v {
			strs = append(strs, fmt.Sprintf("%0.1f元", v1))
		}
		str := strings.Join(strs, ", ")

		fmt.Printf("第%v趟的乘车花费: %v\n", i+1, str)
	}
	fmt.Printf("若一个月上班%v天\n", dayNum)
}

func displayAfterCal(notCutPrice, onlyDiscountCost, beforeCutPrice, newCutPrice float32) {
	fmt.Printf("原价：%0.1f \n", notCutPrice)
	fmt.Printf("只打折: %0.1f, 优惠百分比：%0.1f \n", onlyDiscountCost, (1-onlyDiscountCost/notCutPrice)*100)
	fmt.Printf("之前的: %0.1f, 优惠百分比：%0.1f \n", beforeCutPrice, (1-beforeCutPrice/notCutPrice)*100)
	fmt.Printf("现在的: %0.1f, 优惠百分比：%0.1f\n", newCutPrice, (1-newCutPrice/notCutPrice)*100)
}

func getNoCutCost(costsInOneDay [][]float32, dayNum int) float32 {
	var costs float32 = 0.0
	for _, v0 := range costsInOneDay {
		for _, v1 := range v0 {
			costs += v1
		}
	}
	return costs * float32(dayNum)
}

func getOnlyDiscountCost(costsInOneDay [][]float32, dayNum int) float32 {
	var costs float32 = 0.0
	for _, v0 := range costsInOneDay {
		for _, v1 := range v0 {
			v11 := v1 * 0.9
			costs += v11
		}
	}
	return costs * float32(dayNum)
}

func getBeforeCutCost(costsInOneDay [][]float32, dayNum int) float32 {
	var costs float32 = 0.0
	for _, v0 := range costsInOneDay {
		for i, v1 := range v0 {
			v11 := v1 * 0.9
			if i == 1 {
				v11 = v1 - 2
			}
			if v11 < 0 {
				v11 = 0
			}
			costs += v11
		}
	}
	return costs * float32(dayNum)
}

func getNewCutCost(costsInOneDay [][]float32, dayNum int) float32 {
	var cost float32 = 0.0

	for i := 0; i < dayNum; i++ {
		for _, v0 := range costsInOneDay {
			for i, v1 := range v0 {
				v11 := v1 * getNowDiscount(cost)
				if i == 1 {
					v11 = v11 - 2
				}
				if v11 < 0 {
					v11 = 0
				}
				cost += v11
			}
		}
	}
	return cost
}

func getNowDiscount(cost float32) float32 {
	switch {
	case cost >= 100:
		return 0.5
	case cost >= 50:
		return 0.7
	default:
		return 0.9
	}
}
