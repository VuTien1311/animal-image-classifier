# Manual audit of the most confident test-set "errors"

**Method.** `scripts/error_analysis.py` ranks the fine-tuned ConvNeXt-T's test mistakes by confidence
(`results/figures/errors_convnext_tiny.png`, same order as `label_issues_convnext_tiny.csv` in this
folder). The 20 most confident mistakes were inspected by eye and given one verdict each:

- **label error**: the image clearly shows the predicted class (or no annotated-class animal at all);
- **model error**: the annotated label is right and the model is wrong;
- **ambiguous**: toy, game rendering or extreme close-up; neither label is clearly right.

Verdicts are human judgement, recorded here so they can be checked.

| # | File (under `data/processed/images/test/`) | Label | Predicted | Verdict | What the image shows |
|---:|---|---|---|---|---|
| 1 | `chicken/OIP-kVUa1so_mZk88CZKoILLJwHaFj.jpg` | chicken | dog | label error | hikers with dogs in a forest, no chicken |
| 2 | `cow/OIP-H_CKABzqo3G5HT1SP4bL-gHaEL.jpg` | cow | cat | label error | a tabby cat on a sofa |
| 3 | `horse/OIP-FfUdQmKLrTggr8KSnBcvnQAAAA.jpg` | horse | dog | ambiguous | a plush toy |
| 4 | `cow/OIP-h5tyNnunXV_KlhX-0yN1TAHaE5.jpg` | cow | sheep | label error | a sheep |
| 5 | `cow/OIP-CM9VdkmMdx7T7gYxmDzIjgHaEK.jpg` | cow | chicken | ambiguous | a video-game cow-plant creature |
| 6 | `cow/OIP-kzmcLMcvWDRa7myr16R6sAHaDf.jpg` | cow | sheep | label error | sheep |
| 7 | `cow/OIP-G725phaTAmA_QdNdnqIWTwHaE8.jpg` | cow | horse | label error | a horse |
| 8 | `cow/OIP-mxKJW0v54qe_PLRf-21O6AHaFj.jpg` | cow | horse | label error | a horse |
| 9 | `cow/OIP-yX7rCcMC3JEr6T_Er_u5owHaEK.jpg` | cow | sheep | label error | sheep in a pen |
| 10 | `cow/OIP-dFiMbaJIRR4WOq98z96bfwHaJ4.jpg` | cow | dog | label error | a dog lying on grass |
| 11 | `sheep/OIP-PxFXij5QON7uluG1XtcM_wHaE8.jpg` | sheep | horse | label error | a zebra-horse hybrid |
| 12 | `cat/892.jpg` | cat | dog | model error | a hairless (Cornish Rex) cat |
| 13 | `horse/OIP-h6Hh52_jEoMbHumrAoobgwHaFC.jpg` | horse | cow | model error | horses in a farming-game screenshot |
| 14 | `cow/OIP-Clk22EfdRb623jSiUv3E6wHaEc.jpg` | cow | chicken | label error | a flock of chickens |
| 15 | `cat/99.jpg` | cat | dog | model error | two white kittens |
| 16 | `squirrel/OIP-a1HFPOOx8hCh4TtDk6ChFgHaGX.jpg` | squirrel | sheep | label error | a deer (not one of the 10 classes) |
| 17 | `chicken/OIP-EtQJbgrXIwPsCwRgjGkqmwHaFj.jpg` | chicken | dog | label error | a German shepherd |
| 18 | `sheep/eb37b50f2df0023ed1584d05fb1d4e9fe777ead218ac104497f5c978a6ebb3bf_640.jpg` | sheep | cat | ambiguous | an extreme close-up of a nose |
| 19 | `cat/1898.jpg` | cat | dog | model error | a black-and-white cat |
| 20 | `sheep/OIP-yi_z_1Dcdz1d__KeU-U_7AHaJ2.jpg` | sheep | dog | label error | a shaggy black dog |

**Result.** 13 label errors, 4 model errors, 3 ambiguous. Most confident "mistakes" are mistakes in
the test labels, concentrated in the `cow` folder (8 of the 13). The reported test accuracy is
therefore a slight *under*estimate: correcting only these 13 labels would turn 13 of ConvNeXt-T's
75 errors into correct predictions.
