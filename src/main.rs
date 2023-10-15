use clap::Parser;
use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use rayon::prelude::*;
use std::{collections, error, f64, fs, path, time};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, num_args = 0.., help = "List of files or directories to process")]
    images: Vec<String>,
    #[arg(short, long, help = "Output directory", default_value = "out")]
    out_dir: String,
}

enum GenResult {
    Dir(fs::ReadDir, Option<Box<GenResult>>),
    Single(Option<Result<path::PathBuf, Box<dyn error::Error + Send + Sync>>>),
}

impl GenResult {
    fn from_meta_path(meta: &fs::Metadata, path: path::PathBuf) -> GenResult {
        if meta.is_dir() {
            fs::read_dir(&path).map_or_else(
                |err| GenResult::Single(Some(Err(err.into()))),
                |rd| GenResult::Dir(rd, None),
            )
        } else if meta.is_file() {
            GenResult::Single(Some(Ok(path)))
        } else {
            GenResult::Single(None)
        }
    }
}

impl From<path::PathBuf> for GenResult {
    fn from(value: path::PathBuf) -> Self {
        match fs::metadata(&value) {
            Ok(meta) => GenResult::from_meta_path(&meta, value),
            Err(err) => GenResult::Single(Some(Err(err.into()))),
        }
    }
}

impl Iterator for GenResult {
    type Item = Result<path::PathBuf, Box<dyn error::Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            GenResult::Dir(reader, container) => {
                if let Some(result) = container.as_deref_mut() {
                    if let Some(res) = result.next() {
                        Some(res)
                    } else {
                        container.take();
                        self.next()
                    }
                } else {
                    reader
                        .next()
                        .map(|res| {
                            let entry = res?;
                            let meta = entry.metadata()?;
                            *container =
                                Some(Box::new(GenResult::from_meta_path(&meta, entry.path())));
                            Ok(())
                        })
                        .and_then(|res| match res {
                            Ok(_) => self.next(),
                            Err(err) => Some(Err(err)),
                        })
                }
            }
            GenResult::Single(val) => val.take(),
        }
    }
}

const MAX_BLACK: u8 = 40;
const MAX_GRAY: u8 = 60;

fn main() -> Result<(), Box<dyn Send + Sync + error::Error>> {
    let args = Args::parse();
    let root_path = path::Path::new(&args.out_dir);
    let start_time = time::Instant::now();
    let completion = args
        .images
        .par_iter()
        .flat_map(|path| {
            let base = path::PathBuf::from(path);
            let res: GenResult = base.clone().into();
            let is_dir = matches!(res, GenResult::Dir(_, _));
            res.map(move |p| {
                p.and_then(|rp| {
                    let rel = if is_dir {
                        rp.strip_prefix(&base)?.to_owned()
                    } else {
                        rp.file_name()
                            .ok_or::<Box<dyn Send + Sync + error::Error>>(
                                "Failed to extract file name".into(),
                            )?
                            .into()
                    };
                    Ok((rp, rel))
                })
            })
            .par_bridge()
        })
        .map(|prev| {
            let (realpath, rel) = prev?;
            let img = ImageReader::open(&realpath)?.decode()?;
            let img = img.grayscale();
            let img = img.as_luma8().ok_or("Expected 8 bit grayscale image")?;
            let mut out = ImageBuffer::<Luma<u8>, Vec<u8>>::new(img.width(), img.height());
            let mut vis = vec![false; (img.width() * img.height()) as usize];
            let mut blobs = Vec::new();
            for (x, y, &px) in img.enumerate_pixels() {
                if vis[(y * img.width() + x) as usize] || px.0[0] > MAX_BLACK {
                    continue;
                }
                let mut q = collections::LinkedList::new();
                let mut blob = vec![(x, y)];
                q.push_back((x, y));
                while let Some((x, y)) = q.pop_front() {
                    if vis[(y * img.width() + x) as usize] {
                        continue;
                    }
                    vis[(y * img.width() + x) as usize] = true;
                    blob.push((x, y));
                    for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                        let (cx, cy) = (x.wrapping_add_signed(dx), y.wrapping_add_signed(dy));
                        if cx >= img.width()
                            || cy >= img.height()
                            || vis[(cy * img.width() + cx) as usize]
                            || img[(cx, cy)].0[0] > MAX_GRAY
                        {
                            continue;
                        }
                        q.push_back((cx, cy));
                    }
                }
                if blob.len() > 1 {
                    blobs.push(blob);
                }
            }
            for blob in blobs {
                if blob.len() < 3 || blob.len() > 10000 {
                    continue;
                }
                let (sx, sy) = blob
                    .iter()
                    .copied()
                    .map(|(x, y)| (x as f64, y as f64))
                    .fold((0.0, 0.0), |(ax, ay), (bx, by)| (ax + bx, ay + by));
                let (cx, cy) = (sx / blob.len() as f64, sy / blob.len() as f64);
                let expected_radius = blob.len() as f64 / f64::consts::PI;
                let allowed_radius = expected_radius * 1.5;
                if blob
                    .iter()
                    .copied()
                    .any(|(x, y)| (x as f64 - cx).hypot(y as f64 - cy) > allowed_radius)
                {
                    continue;
                }
                for pt in blob {
                    out[pt] = [255].into();
                }
            }
            image::imageops::colorops::invert(&mut out);
            let target = root_path.join(rel);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            out.save(&target)?;
            Ok((realpath, target))
        })
        .inspect(|res| {
            if let Ok((realpath, target)) = res {
                println!("Segmented {} -> {}", realpath.display(), target.display());
            }
        })
        .map(|res| res.map(|(_, target)| target))
        .collect::<Vec<Result<path::PathBuf, Box<dyn Send + Sync + error::Error>>>>();

    let end_time = time::Instant::now();
    let delta_t = end_time - start_time;
    println!(
        "Processed {} images in {:.3}s ({} errors)",
        completion.iter().filter(|v| v.is_ok()).count(),
        delta_t.as_secs_f64(),
        completion.iter().filter(|v| v.is_err()).count()
    );
    Ok(())
}
