he_folder_path = "C:/Users/beiliu/Desktop/Exposure/HE_testing/";
le_folder_path = "C:/Users/beiliu/Desktop/Exposure/LE_testing/";
pred_folder_path = "C:/Users/beiliu/Desktop/Exposure/Pred_LE_HE/";
samples = getFileList(he_folder_path);


//List.set("Name", "score");
Table.create("Result"); 
r = 0;
for (i = 0; i < samples.length; i++){ //samples.length
	he_imgs = getFileList(he_folder_path + samples[i]);
	avg_le = 0;
	avg_pred = 0;
	for (j = 0; j < 15; j++ ){ //15
		
		// Open Two Images 
		open(he_folder_path + samples[i] + he_imgs[j]); //HE
		open(le_folder_path + samples[i] + replace(he_imgs[j], "HE", "LE")); //LE

		
		pred = replace(he_imgs[j], ".tif", "-1.tif");
		le = replace(he_imgs[j], "HE", "LE");
		gt = he_imgs[j];
		
		run("Calculate Error-Map, RSE and RSP", "reference=&gt super-resolution=&le rsf=[-- RSF unknown, estimate via optimisation --] max.=5");
		selectWindow("RSP and RSE values");
		score = getResult("RSP (Resolution Scaled Pearson-Correlation)", 0);
		//Close windows
		close("*");

		avg_le += score;
		
		selectWindow("Result");
		Table.set("Sample", r, samples[i]);
		Table.set("Image", r, he_imgs[j]);
		Table.set("RSP-LE", r, score);
		
		open(he_folder_path + samples[i] + he_imgs[j]); //HE
		open(pred_folder_path + replace(samples[i],"/", "_pred/" ) + he_imgs[j]); //Pred
		run("Calculate Error-Map, RSE and RSP", "reference=&gt super-resolution=&pred rsf=[-- RSF unknown, estimate via optimisation --] max.=5");
		selectWindow("RSP and RSE values");
		score2 = getResult("RSP (Resolution Scaled Pearson-Correlation)", 0);
		close("*");
	

		avg_pred += score2;
		selectWindow("Result");
		Table.set("RSP-Pred", r, score2);
		Table.update;
		r++;
		
	}
	avg_pred = avg_pred/15;
	avg_le = avg_le/15;
	
	selectWindow("Result");
	Table.set("Sample", r, samples[i] + "Average");
	Table.set("RSP-LE", r, avg_le);
	Table.set("RSP-Pred", r, avg_pred);
	Table.update;
	r++;	
}



