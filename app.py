import os
import io
import json
import shutil
import subprocess
import trimesh
from PIL import Image
from rembg import remove
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "web_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


def make_subject_image(input_path, output_path):
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(input_bytes)
    image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.save(output_path)


def find_first_file_by_ext(folder_path, ext):
    ext = ext.lower()
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(ext):
                return os.path.join(root, f)
    return None


def load_mesh_safely(obj_path):
    loaded = trimesh.load(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0 and len(g.faces) > 0:
                meshes.append(g)
        if not meshes:
            raise ValueError("OBJ 里没有可用网格。")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError("网格为空。")

    return mesh


def build_pointcloud_json(obj_path, json_path, sample_count=22000):
    mesh = load_mesh_safely(obj_path)

    points, _ = trimesh.sample.sample_surface(mesh, sample_count)

    bounds = mesh.bounds
    min_bound = bounds[0].tolist()
    max_bound = bounds[1].tolist()

    data = {
        "points": points.tolist(),
        "min_bound": min_bound,
        "max_bound": max_bound,
        "count": int(len(points)),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        status="等待上传图片...",
        error=None,
        original_url=None,
        subject_url=None,
        pointcloud_url=None
    )


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return render_template(
            "index.html",
            status="没有检测到上传文件。",
            error="没有检测到上传文件。",
            original_url=None,
            subject_url=None,
            pointcloud_url=None
        )

    file = request.files["image"]
    if file.filename == "":
        return render_template(
            "index.html",
            status="你还没有选择图片。",
            error="你还没有选择图片。",
            original_url=None,
            subject_url=None,
            pointcloud_url=None
        )

    clear_folder(UPLOAD_DIR)
    clear_folder(OUTPUT_DIR)

    original_path = os.path.join(UPLOAD_DIR, "original.png")
    subject_path = os.path.join(OUTPUT_DIR, "subject.png")
    pointcloud_path = os.path.join(OUTPUT_DIR, "pointcloud.json")

    file.save(original_path)

    try:
        # 1. 抠主体
        make_subject_image(original_path, subject_path)

        # 2. 跑 TripoSR
        cmd = [
            "python",
            "run.py",
            subject_path,
            "--output-dir",
            OUTPUT_DIR,
            "--mc-resolution",
            "96"
        ]

        subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )

        # 3. 找 OBJ
        obj_path = find_first_file_by_ext(OUTPUT_DIR, ".obj")
        if obj_path is None:
            debug_files = []
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for f in files:
                    debug_files.append(os.path.relpath(os.path.join(root, f), OUTPUT_DIR))

            return render_template(
                "index.html",
                status="主体图已生成，但没有找到 OBJ。",
                error="没有找到 OBJ 文件。\n\n输出目录实际文件：\n" + "\n".join(debug_files),
                original_url="/asset/original.png",
                subject_url="/asset/subject.png",
                pointcloud_url=None
            )

        # 4. 从 TripoSR 的表面生成粒子点云
        build_pointcloud_json(obj_path, pointcloud_path, sample_count=22000)

        return render_template(
            "index.html",
            status="已完成：原图上传、主体抠图、TripoSR建模、烟花粒子生成。",
            error=None,
            original_url="/asset/original.png",
            subject_url="/asset/subject.png",
            pointcloud_url="/output/pointcloud.json"
        )

    except subprocess.CalledProcessError as e:
        err_text = (e.stderr or "") + "\n" + (e.stdout or "")
        return render_template(
            "index.html",
            status="TripoSR 建模失败。",
            error=err_text.strip() if err_text.strip() else "run.py 执行失败",
            original_url="/asset/original.png" if os.path.exists(original_path) else None,
            subject_url="/asset/subject.png" if os.path.exists(subject_path) else None,
            pointcloud_url=None
        )
    except Exception as e:
        return render_template(
            "index.html",
            status="处理失败。",
            error=str(e),
            original_url="/asset/original.png" if os.path.exists(original_path) else None,
            subject_url="/asset/subject.png" if os.path.exists(subject_path) else None,
            pointcloud_url=None
        )


@app.route("/asset/<path:filename>")
def serve_asset(filename):
    if filename == "original.png":
        return send_from_directory(UPLOAD_DIR, filename)
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)