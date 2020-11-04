from add_and_show import readHDF5
import numpy as np
import h5py
import ROOT as rt

def creatTH1F(bins, x_range, title = '', line_color=1):
	h = rt.TH1F('h', title, bins, x_range[0], x_range[1])

	if line_color:
		h.SetLineColor(line_color)
	return h

def normalizeTH1F(h, norm=1):
	scale = norm / h.Integral()
	h.Scale(scale)

	return h

def createLegend(d, position=[0.7,0.8,0.9,0.9]):
	print("createLegend: creating legend...")

	legend = rt.TLegend(position[0], position[1], position[2], position[3])

	for key in d.keys():
		legend.AddEntry(key, d[key])

	return legend



if __name__ == "__main__":
	pos_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/pos/positron_skim_dataset_batch0_N5000.h5"
	u238_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/u238/u238_skim_dataset_batch0_N5000.h5"
	th232_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/Acrylic_Th232/Acrylic_Th232_dataset_batch0_N5000.h5"

	pos = readHDF5(pos_path)
	u238 = readHDF5(u238_path)
	th232 = readHDF5(th232_path)

	pos_totalPE = [sum([sum(x) for x in event]) for event in list(pos)]
	u238_totalPE = [sum([sum(x) for x in event]) for event in list(u238)]
	th232_totalPE = [sum([sum(x) for x in event]) for event in list(th232)]

	x_range = [min(pos_totalPE+u238_totalPE+th232_totalPE), max(pos_totalPE+u238_totalPE+th232_totalPE)]

	c = rt.TCanvas('c1', 'signal and totalPE hist', 200, 10, 600, 400)
	rt.gStyle.SetOptStat(0)
	print(x_range)

	bins = 100
	h_pos = creatTH1F(bins, x_range, title="totalPE_stat", line_color=1)
	h_u238 = creatTH1F(bins, x_range, line_color=2)
	h_th232 = creatTH1F(bins, x_range, line_color=3)
	

	for i in range(len(pos_totalPE)):
		h_pos.Fill(pos_totalPE[i])
		h_u238.Fill(u238_totalPE[i])
		h_th232.Fill(th232_totalPE[i])

	h_pos = normalizeTH1F(h_pos)
	h_u238 = normalizeTH1F(h_u238)
	h_th232 = normalizeTH1F(h_th232)

	h_pos.Draw()
	h_u238.Draw("SAME")
	h_th232.Draw("SAME")

	legend = createLegend({h_pos:"positron", h_u238:"U238", h_th232:"TH232"})
	legend.Draw("SAME")

	c.SaveAs("totalPE_hist.png")
	c.SaveAs("totalPE_hist_norm.png")

