import os
from docling.document_converter import DocumentConverter

def extract_document_elements(document):
    li=[]
    li2=[]
    output_folder = "found_img"
    os.makedirs(output_folder, exist_ok=True)
    if document.texts:
        for text_item in document.texts:
            if text_item.prov:
                prov = text_item.prov[0]
                text = text_item.text
                bbox = prov.bbox
                li2.append({"page_no": prov.page_no-1,
                            "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b},
                            "text":text_item.text})

    if document.pictures:
        for pic_item in document.pictures:
            if pic_item.prov:
                prov = pic_item.prov[0]
                bbox = prov.bbox

                li.append({"page_no":prov.page_no-1,
                           "bbox":{"l":bbox.l,"t":bbox.t,"r":bbox.r,"b":bbox.b},
                           })
                if pic_item.captions:
                    print(f"  Captions: {[c.text for c in pic_item.captions]}")
                if pic_item.children:
                    print("  Associated Text (children of this image):")
                    for child_ref in pic_item.children:
                        found_text = None
                        for t_item in document.texts:
                            if t_item.self_ref == child_ref.cref:
                                found_text = t_item
                                break
                        if found_text and found_text.prov:
                            child_prov = found_text.prov[0]
                            li2.append({"page_no": child_prov.page_no - 1,
                                        "bbox": {"l": child_prov.bbox.l, "t": child_prov.bbox.t, "r": child_prov.bbox.r, "b": child_prov.bbox.b},
                                        "text": found_text.text})

    return li,li2




