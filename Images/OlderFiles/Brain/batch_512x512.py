#!/usr/bin/env python3
"""
Tüm beyin resimlerini 512x512 klasörlerinden tek bir destinasyona taşır
512_raw_ prefix'i ile isimlendirir
"""
import shutil
from pathlib import Path

def batch_move_brain_images():
    """
    Tüm beyin resimlerini 512x512 klasörlerinden tek bir destinasyona taşır
    """
    brain_dir = Path("/home/ubuntu/RadiovLLMjection/Images/Kaggle/SelectedFiles")
    output_dir = Path("/home/ubuntu/RadiovLLMjection/Images/Brain/raw_512x512")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tüm klasörlerdeki resimleri topla (512Glioma, 512Meningioma, 512Pituitary - tümörlüler)
    target_images = []
    tumor_dirs = ['512Glioma', '512Meningioma', '512Pituitary']

    for tumor_dir in tumor_dirs:
        dir_path = brain_dir / tumor_dir
        if dir_path.exists():
            images = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.jpeg')) + list(dir_path.glob('*.png')) + \
                     list(dir_path.glob('*.JPG')) + list(dir_path.glob('*.JPEG')) + list(dir_path.glob('*.PNG'))
            target_images.extend([(tumor_dir, img) for img in images])

    print(f"📊 Bulunan tümör resim sayısı: {len(target_images)}")

    print("🖼️  512x512 GÖRÜNTÜLERİ TAŞIMA İŞLEMİ BAŞLATILIYOR")
    print("=" * 60)
    print(f"📊 İşlenecek resim sayısı: {len(target_images)}")
    print(f"📁 Kaynak: {brain_dir}")
    print(f"📁 Hedef: {output_dir}")
    print("🎯 Dosya adları: 512_raw_ prefix'i ile")
    
    total_moved = 0
    total_skipped = 0

    for tumor_dir, img_path in target_images:
        img_file = img_path.name

        if not img_path.exists():
            print(f"⚠️  Resim bulunamadı: {tumor_dir}/{img_file}")
            continue

        # Çıktı dosya adı: 512_raw_ prefix'i ile
        output_name = f"512_raw_{img_file}"
        output_path = output_dir / output_name

        # Eğer dosya zaten varsa atla
        if output_path.exists():
            print(f"⏭️  Zaten var, atlanıyor: {output_name}")
            total_skipped += 1
            continue

        try:
            # Dosyayı kopyala (taşı değil, orijinal kalmalı)
            shutil.copy2(img_path, output_path)
            total_moved += 1
            print(f"✅ Taşındı: {output_name}")
        except Exception as e:
            print(f"❌ HATA: {img_file} - {e}")

    print(f"\n🎉 TOPLAM TAŞINAN RESİM: {total_moved}")
    print(f"⏭️  ATLANAN RESİM: {total_skipped}")
    print(f"📁 Hedef klasör: {output_dir}")

    # Örnek çıktı göster
    print("\n📋 ÖRNEK ÇIKTILAR:")
    sample_files = list(output_dir.glob("512_raw_*.jpg")) + \
                   list(output_dir.glob("512_raw_*.jpeg")) + \
                   list(output_dir.glob("512_raw_*.png"))
    
    for i, img_file in enumerate(sample_files[:5]):
        if img_file.exists():
            size = img_file.stat().st_size
            print(f"• {img_file.name}: {size} bytes")

    return output_dir

if __name__ == "__main__":
    print("🖼️  BEYİN GÖRÜNTÜLERİNİ 512x512 KLASÖRLERİNDEN TAŞIMA SİSTEMİ")
    print("=" * 70)

    # Kullanıcı onayını sor
    choice = input("\nTüm beyin resimlerini taşımak istiyor musunuz? (y/n): ").strip().lower()

    if choice == "y":
        output_dir = batch_move_brain_images()
        print("\n🎉 TAŞIMA İŞLEMİ TAMAMLANDI!")
        print(f"📁 Tüm çıktılar: {output_dir}")
    else:
        print("❌ İşlem iptal edildi")

