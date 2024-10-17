import bpy
import os
import mathutils
import json
import argparse
from PIL import Image
import numpy as np

# Function to clamp the image
def clamp_the_image(alpha, with_shadow_array, without_shadow_array):
    shadow_diff = np.abs(with_shadow_array[..., :3] - without_shadow_array[..., :3])
    shadow_mask = np.mean(shadow_diff, axis=-1) * alpha
    shadow_mask = (np.clip(shadow_mask, 0, 1) * 255).astype(np.uint8) 
    return shadow_mask

# Function to calculate shadow mask and reconstruct image
def calculate_shadow_and_reconstruct(original_image, fg_image, fg_image_with_shadow_array, alpha=3.69):
    fg_image_with_shadow_catcher_array = np.asarray(fg_image) / 255.0
    fg_image_mask = fg_image_with_shadow_catcher_array[..., 3] == 1
    shadow_mask_compare = clamp_the_image(1.0, fg_image_with_shadow_catcher_array, fg_image_with_shadow_array)
    shadow_mask_compare = np.stack([shadow_mask_compare] * 3, axis=-1)

    whole_shadow_mask = shadow_mask_compare[..., 0]
    whole_shadow_mask = whole_shadow_mask * fg_image_mask
    bg_mask = 1 - fg_image_mask
    whole_shadow_mask = whole_shadow_mask / 255.0 + fg_image_with_shadow_catcher_array[..., 3] * bg_mask
    whole_shadow_mask = np.stack([whole_shadow_mask] * 3, axis=-1)

    black_shadow = np.zeros_like(whole_shadow_mask)
    clamped_shadow_mask = np.stack([clamp_the_image(alpha, whole_shadow_mask, black_shadow) / 255.0] * 3, axis=-1) * fg_image_mask[..., None] + fg_image_with_shadow_catcher_array[..., 3][..., None] * bg_mask[..., None]

    ref_image_rgb = np.asarray(original_image)[..., :3] / 255.0
    new_image = ref_image_rgb * (1 - clamped_shadow_mask) + black_shadow * clamped_shadow_mask 

    return shadow_mask_compare, clamped_shadow_mask, new_image

class BoundingBoxAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lowest_z = float('inf')
        self.highest_z = float('-inf')
        self.lowest_y = float('inf')
        self.highest_y = float('-inf')
        self.lowest_x = float('inf')
        self.highest_x = float('-inf')

    def process_object(self, obj):
        if obj.bound_box:
            for vertex in obj.bound_box:
                world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
                z_coord = world_vertex.z
                y_coord = world_vertex.y
                x_coord = world_vertex.x
                
                if z_coord < self.lowest_z:
                    self.lowest_z = z_coord
                if z_coord > self.highest_z:
                    self.highest_z = z_coord
                
                if y_coord < self.lowest_y:
                    self.lowest_y = y_coord
                if y_coord > self.highest_y:
                    self.highest_y = y_coord
                
                if x_coord < self.lowest_x:
                    self.lowest_x = x_coord
                if x_coord > self.highest_x:
                    self.highest_x = x_coord

    def traverse_hierarchy(self, obj):
        if not obj.children:
            self.process_object(obj)
        else:
            for child in obj.children:
                self.traverse_hierarchy(child)

    def analyze(self, root_parent):
        self.traverse_hierarchy(root_parent)

    @staticmethod
    def get_root_parent(obj):
        while obj.parent:
            obj = obj.parent
        return obj


def main():
    # Paths to assets

    parser = argparse.ArgumentParser(description='File paths for rendering')

    parser.add_argument('--root_dir', type=str, default='./', help='Root directory')
    parser.add_argument('--car_path', type=str, default='./assets/d5e474d0cecf4688b488d84819dc8790.glb', help='Path to car model')
    parser.add_argument('--trajectory_path', type=str, default='./assets/bezier_curves.json', help='Path to trajectory file')
    parser.add_argument('--bg_image_path', type=str, default='./assets/000018_0.png', help='Path to background image')
    parser.add_argument('--env_map_path', type=str, default='./assets/000018_0.exr', help='Path to environment map')
    parser.add_argument('--mesh_map_path', type=str, default='./assets/transformed_mesh_center_25_0_0_extents_50_50_50.obj', help='Path to mesh map')

    args = parser.parse_args()

    root_dir = args.root_dir
    car_path = args.root_dir + args.car_path
    trajectory_path = args.root_dir + args.trajectory_path
    bg_image_path = args.root_dir + args.bg_image_path
    env_map_path = args.root_dir + args.env_map_path
    mesh_map_path = args.root_dir + args.mesh_map_path

    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Import the car model
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=car_path)

    print("GLB file imported and car object selected successfully.")

    root_parents = set()
    for obj in bpy.context.selected_objects:
        root_parent = BoundingBoxAnalyzer.get_root_parent(obj)
        root_parents.add(root_parent)

    analyzer_list = []

    for root_parent in root_parents:
        print(f"Analysis for root parent {root_parent.name}:")
        root_parent.scale = (1, 1, 1)
        root_parent.rotation_quaternion = mathutils.Quaternion((0.5, -0.5, 0.5, -0.5))
        root_parent.location = (0, 0, 0)
        bpy.context.view_layer.update()

        analyzer = BoundingBoxAnalyzer()
        analyzer.analyze(root_parent)
        print("Lowest Z value of the bounding box in world coordinates:", analyzer.lowest_z)
        print("Highest Z value of the bounding box in world coordinates:", analyzer.highest_z)
        print("Lowest Y value of the bounding box in world coordinates:", analyzer.lowest_y)
        print("Highest Y value of the bounding box in world coordinates:", analyzer.highest_y)
        print("Lowest X value of the bounding box in world coordinates:", analyzer.lowest_x)
        print("Highest X value of the bounding box in world coordinates:", analyzer.highest_x)
        analyzer_list.append(analyzer)

        height = analyzer.highest_y - analyzer.lowest_y
        reference_height = 2.0
        scale_factor = reference_height / height

        print("Scale factor:", scale_factor)
        new_x = (analyzer.highest_x + analyzer.lowest_x) / 2
        new_y = (analyzer.highest_y + analyzer.lowest_y) / 2
        # new_z = -analyzer.lowest_z * scale_factor
        new_z = -1.9369
        print("New location:", new_x, new_y, new_z)

        root_parent.location = (new_x, new_y, new_z)
        root_parent.scale *= scale_factor
        bpy.context.view_layer.update()

        analyzer.reset()
        analyzer.analyze(root_parent)
        print("After adjustment:")
        print("Lowest Z value of the bounding box in world coordinates:", analyzer.lowest_z)
        print("Highest Z value of the bounding box in world coordinates:", analyzer.highest_z)
        print("Lowest Y value of the bounding box in world coordinates:", analyzer.lowest_y)
        print("Highest Y value of the bounding box in world coordinates:", analyzer.highest_y)
        print("Lowest X value of the bounding box in world coordinates:", analyzer.lowest_x)
        print("Highest X value of the bounding box in world coordinates:", analyzer.highest_x)
        
    print("Analysis complete.")

    # Set the environment map
    env_map = bpy.data.images.load(env_map_path)

    # Ensure the world uses nodes
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add Background node
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.location = 0, 0

    # Add Environment Texture node
    env_text_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_text_node.image = env_map
    env_text_node.location = -300, 0

    # Add Output node
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = 200, 0

    # Link nodes
    links.new(env_text_node.outputs['Color'], background_node.inputs['Color'])
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    # Set the strength of the background node
    background_node.inputs['Strength'].default_value = 3.0

    # Update the scene to ensure the environment map is applied
    bpy.context.view_layer.update()

    print("Environment map set successfully.")

    # Import the trajectory
    with open(trajectory_path, 'r') as file:
        data = json.load(file)

    curve_data = bpy.data.curves.new(name='BezierCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12

    spline = curve_data.splines.new(type='BEZIER')
    spline.bezier_points.add(len(data[0]['points']) - 1)

    for i, point in enumerate(data[0]['points']):
        bezier_point = spline.bezier_points[i]
        bezier_point.co = point['co']
        bezier_point.handle_left = point['handle_left']
        bezier_point.handle_right = point['handle_right']
        bezier_point.radius = point['radius']
        bezier_point.tilt = point['tilt']

    curve_object = bpy.data.objects.new('BezierCurveObject', curve_data)
    # tune in the blender UI
    curve_object.location = (35.671, -1.48526, 0)
    bpy.context.collection.objects.link(curve_object)

    print("Trajectory processing complete.")

    start_frame = 1
    end_frame = 40

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    car_object = bpy.data.objects.get(root_parent.name)

    follow_path_constraint = car_object.constraints.new(type='FOLLOW_PATH')
    follow_path_constraint.target = curve_object
    follow_path_constraint.use_curve_follow = True
    follow_path_constraint.forward_axis = 'TRACK_NEGATIVE_X'

    curve_object.data.path_duration = end_frame
    curve_object.data.use_path = True

    curve_object.data.eval_time = 0
    curve_object.data.keyframe_insert(data_path='eval_time', frame=start_frame)

    curve_object.data.eval_time = end_frame - 1
    curve_object.data.keyframe_insert(data_path='eval_time', frame=end_frame)    

    print("Car animation along the path set successfully.")

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.device = 'GPU'

    # Load the mesh from mesh_map_path
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.wm.obj_import(filepath=mesh_map_path)
    mesh_map = bpy.context.selected_objects[0]

    # Set the location, rotation, and scale of the imported mesh
    mesh_map.location = (0, 0, 0)
    mesh_map.rotation_mode = 'QUATERNION'
    mesh_map.rotation_quaternion = (1, 0, 0, 0)
    mesh_map.scale = (1, 1, 1)
    bpy.ops.object.select_all(action='DESELECT')

    print("Mesh loaded and placed successfully.")


    camera_location = (0, 0, 0)
    camera_quaternion = (0.5, 0.5, -0.5, -0.5)
    camera_fov = 0.879646

    bpy.ops.object.camera_add(location=camera_location)
    camera_object = bpy.context.object
    camera_object.rotation_mode = 'QUATERNION'
    camera_object.rotation_quaternion = camera_quaternion
    camera_object.data.lens_unit = 'FOV'
    camera_object.data.angle = camera_fov

    bg_image = bpy.data.images.load(bg_image_path)
    height = bg_image.size[1]
    width = bg_image.size[0]
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height

    camera_object.data.show_background_images = True

    bg = camera_object.data.background_images.new()
    bg.image = bg_image
    bg.alpha = 1.0

    bpy.context.scene.camera = camera_object

    print("Camera added successfully.")

    bpy.context.scene.cycles.max_bounces = 5
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.film_transparent_glass = True

    for frame in range(start_frame, end_frame + 1):
        output_path = f"./cache/rendered_image_shadow_catcher_{frame}.png"
        if os.path.exists(output_path):
            print(f"Frame {frame} already rendered. Skipping.")
            continue

        if frame > 3:
            break

        bpy.context.scene.frame_set(frame)
        curve_object.data.eval_time = frame
        curve_object.data.keyframe_insert(data_path='eval_time', frame=frame)
        bpy.context.view_layer.update()
        
        # Deselect all objects and select mesh_map
        bpy.ops.object.select_all(action='DESELECT')
        mesh_map.select_set(True)
        
        # First render with mesh_map as shadow catcher
        mesh_map.is_shadow_catcher = True
        mesh_map.visible_shadow = False
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        output_path = f"./cache/rendered_image_shadow_catcher_{frame}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bpy.data.images['Render Result'].save_render(filepath=output_path)
        
        # Second render without mesh_map as shadow catcher
        mesh_map.is_shadow_catcher = False
        mesh_map.visible_shadow = True
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        with_car_shadow_path = f"./cache/rendered_image_car_with_shadow_{frame}.png"
        bpy.data.images['Render Result'].save_render(filepath=with_car_shadow_path)
        
        print(f"Frame {frame} rendered successfully.")



    print("Rendering complete.")


    input_shadow_catcher_template = 'cache/rendered_image_shadow_catcher_{}.png'
    input_car_with_shadow_template = 'cache/rendered_image_car_with_shadow_{}.png'
    output_composite_template = 'results/composite_image_{}.png'
    output_composite_side_by_side_template = 'results/composite_side_by_side_{}.png'

    # Load the original image
    original_image = Image.open(bg_image_path).convert("RGBA")

    # Loop through the images and generate the composite images
    for i in range(1, 41):
        shadow_catcher_path = input_shadow_catcher_template.format(i)
        car_with_shadow_path = input_car_with_shadow_template.format(i)
        output_path = output_composite_template.format(i)

        # Load the foreground images
        fg_image = Image.open(shadow_catcher_path).convert("RGBA")
        fg_image_with_shadow = Image.open(car_with_shadow_path).convert("RGBA")
        # composite original_image and fg_image
        ref_composited_image = Image.alpha_composite(original_image, fg_image)
        fg_image_with_shadow_array = np.asarray(fg_image_with_shadow) / 255.0

        # Calculate shadow mask and reconstruct image
        _, whole_shadow_mask, new_image = calculate_shadow_and_reconstruct(ref_composited_image, fg_image, fg_image_with_shadow_array)

        # Save the composite image
        new_image = (new_image * 255).astype(np.uint8)
        new_image_pil = Image.fromarray(new_image)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_image_pil.save(output_path)

        # save the whole shadow mask
        whole_shadow_mask = (whole_shadow_mask * 255).astype(np.uint8)
        whole_shadow_mask_pil = Image.fromarray(whole_shadow_mask)
        whole_shadow_mask_pil.save(f"cache/whole_shadow_mask_{i}.png")

        print(f"Composite image {i} saved successfully.")
        harmonization_mask = (whole_shadow_mask[..., 0] > 0).astype(np.uint8) * 255
        harmonization_mask_pil = Image.fromarray(harmonization_mask)
        harmonization_mask_pil.save(f"cache/harmonization_mask_{i}.png")

        # concate new_image_pil, fg_image_with_shadow, whole_shadow_mask side by side and save the image
        new_image_pil = new_image_pil.convert("RGBA")
        fg_image_with_shadow = fg_image_with_shadow.convert("RGBA")
        whole_shadow_mask_pil = whole_shadow_mask_pil.convert("RGBA")

        # Create a new image with width = sum of widths and height = max of heights
        width = new_image_pil.width + fg_image_with_shadow.width + whole_shadow_mask_pil.width
        height = max(new_image_pil.height, fg_image_with_shadow.height, whole_shadow_mask_pil.height)
        concatenated_image = Image.new('RGBA', (width, height))

        # Paste images side by side
        concatenated_image.paste(new_image_pil, (0, 0))
        concatenated_image.paste(fg_image_with_shadow, (new_image_pil.width, 0))
        concatenated_image.paste(whole_shadow_mask_pil, (new_image_pil.width + fg_image_with_shadow.width, 0))

        # Save the concatenated image
        concatenated_image.save(output_composite_side_by_side_template.format(i))

        print(f"Concatenated side-by-side image {i} saved successfully.")

if __name__ == "__main__":
    main()