import os

def save_filenames_to_txt(directory, output_file):
    try:
        with open(output_file, 'w') as f:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')
        print(f"File names have been successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
directory = '/home/cn/Datasets/SakugaDataset/split/train_full_81'  # 替换为你的文件夹路径
output_file = '/home/cn/Datasets/SakugaDataset/output_81.txt'  # 替换为你想保存文件名的文本文件路径

save_filenames_to_txt(directory, output_file)
