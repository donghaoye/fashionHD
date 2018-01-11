from __future__ import division, print_function

import os
import util.io as io
import numpy as np


def output_attribute_entry():
    '''
    Output formated attribute entry to txt file.
    '''

    attr_entry = io.load_json('datasets/DeepFashion/In-shop/Label/attribute_entry_top500.json')
    type_map = {
        0: 'Color',
        1: 'Texture',
        2: 'Fabric',
        3: 'Shape',
        4: 'Part',
        5: 'Style'
    }

    # output attribute entry txt
    attr_entry_list = []

    for att_type, type_name in sorted(type_map.items()):
        attr_this_type = [att for att in attr_entry if att['type'] == att_type]
        attr_this_type.sort(key = lambda x:x['pos_rate'], reverse = True)
        attr_entry_list += attr_this_type


    attr_txt_list = ['%s\t%s(%d)\t%.3f' % (att['entry'], type_map[att['type']], att['type'], att['pos_rate'])
        for att in attr_entry_list]
    attr_name_list =[att['entry'] for att in attr_entry_list]

    dir_out = 'temp/attribute_entry/'
    io.mkdir_if_missing(dir_out)
    io.save_str_list(attr_txt_list, os.path.join(dir_out, 'attribute_entry.txt'))
    io.save_str_list(attr_name_list, os.path.join(dir_out, 'attribute_entry_name.txt'))

    io.save_json(attr_entry_list, 'datasets/DeepFashion/In-shop/Label/attribute_entry_top500_byType.json')


    # output positive samples for each attribute entry
    num_example = 10
    samples = io.load_json('datasets/DeepFashion/In-shop/Label/samples_attr.json')
    attr_label = io.load_json('datasets/DeepFashion/In-shop/Label/attribute_label_top500.json')
    items ={}

    for s_id, s in samples.iteritems():
        items[s['item_id']] = {
            'id': s_id,
            'item_id': s['item_id'],
            'img_path': s['img_path'],
            'label': attr_label[s_id]
        }

    item_list = items.values()

    dir_example = os.path.join(dir_out, 'examples')
    io.mkdir_if_missing(dir_example)

    for idx, att in enumerate(attr_entry):
        print('search examples for attribute %d / %d: %s' % (idx, len(attr_entry), att['entry']))
        dir_example_this_att = os.path.join(dir_example, att['entry'].replace(' ', '_'))
        io.mkdir_if_missing(dir_example_this_att)
        pos_list = [item for item in item_list if item['label'][idx] == 1]
        np.random.shuffle(pos_list)

        for item in pos_list[0:num_example]:
            fn_src = item['img_path']
            fn_tar = os.path.join(dir_example_this_att, item['item_id']+'.jpg')
            io.copy(fn_src, fn_tar)



def attribute_fusion_retrieval():
    
    upper_categorys = {'Tees_Tanks', 'Blouses_Shirts', 'Sweaters',
        'Jackets_Coats', 'Graphic_Tees', 'Sweatshirts_Hoodies', 'Cardigans',
        'Shirts_Polos', 'Jackets_Vests'}

    type_map = {
        0: 'Color',
        1: 'Texture',
        2: 'Fabric',
        3: 'Shape',
        4: 'Part',
        5: 'Style'
    }

    samples = io.load_json('datasets/DeepFashion/In-shop/Label/samples_attr.json')
    attr_label = io.load_json('datasets/DeepFashion/In-shop/Label/attribute_label_top500.json')
    attr_entry = io.load_json('datasets/DeepFashion/In-shop/Label/attribute_entry_top500.json')

    samples = {s_id: s for s_id, s in samples.iteritems() if s['category'] in upper_categorys and s['pose'] == 'front'}
    print('valid images: %d' % len(samples))

    items = {}
    for s_id, s in samples.iteritems():
        items[s['item_id']] = {
            'id': s_id,
            'item_id': s['item_id'],
            'img_path': s['img_path'],
            'label': attr_label[s_id]
        }
    print('valid items: %d' % len(items))


    id_list = [it['id'] for it in items.values()]
    attr_mat = np.array([attr_label[s_id] for s_id in id_list], dtype = np.float32)

    attr_indices_list = []
    attr_group = []

    i = 0
    for att_type in range(6):
        type_indices = [idx for idx, att in enumerate(attr_entry) if att['type'] == att_type]
        attr_indices_list += type_indices
        
        type_size = len(type_indices)
        attr_group.append({
            'type_name': type_map[att_type],
            'i_start': i,
            'i_end': i+type_size,
            })
        i += type_size

        print('type [%s]: %d attributes' % (type_map[att_type], type_size))

    
    attr_mat = attr_mat[:, attr_indices_list]
    attr_entry = [attr_entry[i] for i in attr_indices_list]
    
    for ag in attr_group:
        attr_mat[:, ag['i_start']: ag['i_end']] /= (ag['i_end']-ag['i_start'])
    
    
    # perform retrieval and return results

    def _get_attr_str(label):
        label = label.tolist()
        attr_str = ['%s[%s]' % (att['entry'], type_map[att['type']]) for i, att in enumerate(attr_entry) if label[i] > 0]
        return ' '.join(attr_str)

    N = 100
    src_idx_list = np.random.choice(range(len(id_list)), N).tolist()
    tar_idx_list = np.random.choice(range(len(id_list)), N).tolist()

    for n, (src_idx, tar_idx) in enumerate(zip(src_idx_list, tar_idx_list)):
        dir_out = 'temp/attribute_retrieval/%d' % n
        io.mkdir_if_missing(dir_out)
        txt_list = []

        src_id = id_list[src_idx]
        tar_id = id_list[tar_idx]
        io.copy(samples[src_id]['img_path'], os.path.join(dir_out, 'input_src.jpg'))
        io.copy(samples[tar_id]['img_path'], os.path.join(dir_out, 'input_tar.jpg'))

        v_src = attr_mat[src_idx,:]
        v_tar = attr_mat[tar_idx,:]

        txt_list.append('src:\t %s' % _get_attr_str(v_src))
        txt_list.append('')
        txt_list.append('tar:\t %s' % _get_attr_str(v_tar))
        txt_list.append('')
        txt_list.append('')

        for att_type in range(6):
            ag = attr_group[att_type]
            v_query = v_src.copy()
            v_query[ag['i_start']:ag['i_end']] = v_tar[ag['i_start']:ag['i_end']]
            dist = np.abs(attr_mat - v_query).sum(axis = 1)

            rtv_idx = dist.argsort()[1]
            rtv_id = id_list[rtv_idx]
            v_rtv = attr_mat[rtv_idx,:]

            io.copy(samples[rtv_id]['img_path'], os.path.join(dir_out, 'retrieve_%d_%s.jpg' % (att_type, type_map[att_type])))
            txt_list.append('retrieve [%s]: %s' % (type_map[att_type], _get_attr_str(v_rtv)))

        io.save_str_list(txt_list, os.path.join(dir_out, 'info.txt'))
        txt_list.append('')







if __name__ == '__main__':
    # output_attribute_entry()
    attribute_fusion_retrieval()
