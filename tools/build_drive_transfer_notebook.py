"""Build a standalone Colab notebook for personal Drive ownership transfers."""

import json
from pathlib import Path
from textwrap import dedent


cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": dedent(source).strip() + "\n", "execution_count": None, "outputs": []})


markdown("""
    # Chuyển quyền sở hữu Google Drive

    Dành cho hai tài khoản Gmail cá nhân. Quét toàn bộ thư mục con, giữ nguyên file và cấu trúc.

    **Trước khi chạy:** dừng notebook training sau khi checkpoint đã lưu. Tài khoản nhận cần đủ dung lượng.

    - **Lần 1 — tài khoản cũ:** chọn `SEND`, nhập hai email, tải danh sách JSON, rồi gửi yêu cầu.
    - **Lần 2 — tài khoản mới:** mở notebook trong phiên Colab mới, chọn `ACCEPT`, tải lên JSON đã lưu, rồi chấp nhận.
    - Chỉ cần CPU. Chạy các cell theo thứ tự. Có thể chạy lại sau khi bị ngắt; file đã xử lý được bỏ qua.
    - Google có thể gửi nhiều email yêu cầu chuyển. Không cần mở từng email khi dùng bước `ACCEPT`.

    Chỉ file được tài khoản mới nhận quyền sở hữu mới giảm dung lượng tài khoản cũ.
    Không theo lối tắt ra ngoài thư mục. File thuộc tài khoản thứ ba được báo riêng.
""")

markdown("""
    ## 00. Chọn chế độ và tài khoản

    Giữ URL nếu chuyển bài DeepWeeds hiện tại. Có thể thay bằng URL thư mục khác.
    Nếu thư mục gốc đã chuyển, vẫn điền **email chủ sở hữu cũ của các file còn lại**.
""")
code('''
    MODE = "SEND"  # Use SEND on the old account and ACCEPT on the new account.
    ROOT_FOLDER_URL = "https://drive.google.com/drive/folders/1OJHKQBicCgrSfG42pETh5xvBsohBXVKA"
    SOURCE_EMAIL = input("Email tài khoản cũ: ").strip().lower()
    TARGET_EMAIL = input("Email tài khoản mới: ").strip().lower()
''')

markdown("""
    ## 01. Kết nối đúng tài khoản

    `SEND` cần đăng nhập tài khoản cũ; `ACCEPT` cần tài khoản mới.
    Khi đổi tài khoản, dùng phiên Colab mới để tránh giữ thông tin đăng nhập cũ.
""")
code('''
    import json
    import re
    import time
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    import google.auth
    from google.colab import auth, files
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from IPython.display import display
    import pandas as pd

    FOLDER_MIME = "application/vnd.google-apps.folder"
    FILE_FIELDS = "id,name,mimeType,size,quotaBytesUsed,parents,owners(emailAddress),driveId,trashed"
    PERMISSION_FIELDS = "nextPageToken,permissions(id,type,role,emailAddress,pendingOwner)"
    MODE = MODE.strip().upper()
    if MODE not in {"SEND", "ACCEPT"}:
        raise ValueError("MODE phải là SEND hoặc ACCEPT.")
    for email in (SOURCE_EMAIL, TARGET_EMAIL):
        if not re.fullmatch(r"[^@\\s]+@(gmail|googlemail)\\.com", email):
            raise ValueError("Hãy nhập đầy đủ địa chỉ Gmail cá nhân.")
    if SOURCE_EMAIL == TARGET_EMAIL:
        raise ValueError("Hai tài khoản phải khác nhau.")

    match = re.search(r"/folders/([A-Za-z0-9_-]+)", ROOT_FOLDER_URL)
    if not match:
        raise ValueError("URL phải là đường liên kết tới thư mục Drive.")
    ROOT_ID = match.group(1)

    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    account = service.about().get(fields="user(emailAddress,permissionId),storageQuota").execute()
    CURRENT_EMAIL = account["user"]["emailAddress"].lower()
    EXPECTED_EMAIL = SOURCE_EMAIL if MODE == "SEND" else TARGET_EMAIL
    if CURRENT_EMAIL != EXPECTED_EMAIL:
        raise RuntimeError(
            f"Đang kết nối {CURRENT_EMAIL}, nhưng {MODE} cần {EXPECTED_EMAIL}. "
            "Mở phiên Colab mới bằng đúng tài khoản."
        )
    print("Chế độ:", MODE)
    print("Tài khoản đã kết nối:", CURRENT_EMAIL)
    print("Thư mục:", ROOT_FOLDER_URL)
''')

markdown("""
    ## 02. Hàm xử lý

    Chỉ dùng Drive API; không tải dữ liệu ảnh hoặc checkpoint xuống máy.
    Nhật ký ghi sau từng file để xem file lỗi và chạy lại.
""")
code('''
    def api_call(request):
        # Retry transient server and quota failures without retrying permanent errors.
        return request.execute(num_retries=5)


    def metadata(file_id):
        return api_call(service.files().get(fileId=file_id, fields=FILE_FIELDS))


    def owner_email(item):
        owners = item.get("owners", [])
        if len(owners) != 1:
            raise ValueError("File không có đúng một chủ sở hữu cá nhân.")
        return owners[0].get("emailAddress", "").lower()


    def children(folder_id):
        page_token = None
        while True:
            response = api_call(service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                pageSize=1000,
                pageToken=page_token,
                fields=f"nextPageToken,incompleteSearch,files({FILE_FIELDS})",
            ))
            if response.get("incompleteSearch"):
                raise RuntimeError("Drive trả danh sách chưa đầy đủ. Hãy chạy lại bước quét.")
            yield from response.get("files", [])
            page_token = response.get("nextPageToken")
            if not page_token:
                break


    def permission_for(file_id, email):
        page_token = None
        while True:
            response = api_call(service.permissions().list(
                fileId=file_id,
                pageSize=100,
                pageToken=page_token,
                fields=PERMISSION_FIELDS,
            ))
            for permission in response.get("permissions", []):
                if permission.get("type") == "user" and permission.get("emailAddress", "").lower() == email:
                    return permission
            page_token = response.get("nextPageToken")
            if not page_token:
                return None


    def scan_folder():
        root = metadata(ROOT_ID)
        if root.get("trashed") or root.get("driveId") or root["mimeType"] != FOLDER_MIME:
            raise ValueError("Cần một thư mục My Drive còn tồn tại.")
        if owner_email(root) not in {SOURCE_EMAIL, TARGET_EMAIL}:
            raise ValueError("Thư mục gốc không thuộc một trong hai tài khoản đã nhập.")
        records = []
        seen = set()
        queue = [(root, root["name"], 0)]
        while queue:
            item, path, depth = queue.pop()
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            item.update(path=path, depth=depth)
            records.append(item)
            if item["mimeType"] == FOLDER_MIME:
                queue.extend((child, f"{path}/{child['name']}", depth + 1) for child in children(item["id"]))
            if len(records) % 100 == 0:
                print("Đã quét:", len(records))
        return {
            "version": 1,
            "root_id": ROOT_ID,
            "source_email": SOURCE_EMAIL,
            "target_email": TARGET_EMAIL,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "items": records,
        }


    def validate_manifest(manifest):
        if manifest.get("version") != 1:
            raise ValueError("Phiên bản danh sách không hợp lệ.")
        expected = {"root_id": ROOT_ID, "source_email": SOURCE_EMAIL, "target_email": TARGET_EMAIL}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Danh sách không khớp cấu hình: {key}.")
        items = manifest.get("items", [])
        by_id = {item["id"]: item for item in items}
        if not items or len(by_id) != len(items) or ROOT_ID not in by_id:
            raise ValueError("Danh sách trống, trùng ID hoặc thiếu thư mục gốc.")
        if by_id[ROOT_ID]["mimeType"] != FOLDER_MIME:
            raise ValueError("Mục gốc không phải thư mục.")
        connected = {ROOT_ID}
        while True:
            added = {
                item["id"] for item in items
                if any(parent in connected and by_id[parent]["mimeType"] == FOLDER_MIME for parent in item.get("parents", []))
            } - connected
            if not added:
                break
            connected.update(added)
        if connected != set(by_id):
            raise ValueError("Có file nằm ngoài cây thư mục đã chọn.")
        return by_id


    def check_live_item(item, by_id):
        # Recheck the current owner and parent before each permission change.
        current = metadata(item["id"])
        if current.get("trashed") or current.get("driveId"):
            raise ValueError("File đã bị xoá hoặc chuyển vào Shared Drive.")
        if current["mimeType"] != item["mimeType"]:
            raise ValueError("Loại file đã thay đổi.")
        if item["id"] != ROOT_ID:
            original_parents = set(item.get("parents", []))
            if not original_parents.intersection(current.get("parents", [])):
                raise ValueError("File đã chuyển vị trí sau khi quét; cần quét lại.")
            if not any(parent in by_id for parent in current.get("parents", [])):
                raise ValueError("File không còn nằm trong cây thư mục.")
        return current


    def send_request(item, by_id):
        current = check_live_item(item, by_id)
        owner = owner_email(current)
        if owner == TARGET_EMAIL:
            return "ALREADY_TRANSFERRED"
        if owner != SOURCE_EMAIL:
            return "OTHER_OWNER"
        permission = permission_for(item["id"], TARGET_EMAIL)
        if permission and permission.get("pendingOwner"):
            return "ALREADY_PENDING"
        if permission:
            api_call(service.permissions().update(
                fileId=item["id"],
                permissionId=permission["id"],
                body={"role": "writer", "pendingOwner": True},
                fields="id,role,pendingOwner",
            ))
        else:
            api_call(service.permissions().create(
                fileId=item["id"],
                body={"type": "user", "role": "writer", "emailAddress": TARGET_EMAIL, "pendingOwner": True},
                sendNotificationEmail=True,
                fields="id,role,pendingOwner",
            ))
        return "REQUESTED"


    def accept_request(item, by_id):
        current = check_live_item(item, by_id)
        owner = owner_email(current)
        if owner == TARGET_EMAIL:
            return "ALREADY_TRANSFERRED"
        if owner != SOURCE_EMAIL:
            return "OTHER_OWNER"
        permission = permission_for(item["id"], TARGET_EMAIL)
        if not permission or not permission.get("pendingOwner"):
            return "NOT_PENDING"
        # The receiving account explicitly accepts the pending transfer.
        api_call(service.permissions().update(
            fileId=item["id"],
            permissionId=permission["id"],
            body={"role": "owner"},
            transferOwnership=True,
            fields="id,role",
        ))
        if owner_email(metadata(item["id"])) != TARGET_EMAIL:
            raise RuntimeError("Drive chưa xác nhận chủ sở hữu mới. Chạy lại để kiểm tra.")
        return "TRANSFERRED"


    def process_manifest(manifest):
        by_id = validate_manifest(manifest)
        # Transfer files first and folders from the deepest level to the root.
        ordered = sorted(manifest["items"], key=lambda item: (item["mimeType"] == FOLDER_MIME, -item["depth"], item["path"]))
        log_path = Path(f"drive_transfer_{MODE.lower()}_log.jsonl")
        outcomes = []
        action = send_request if MODE == "SEND" else accept_request
        for index, item in enumerate(ordered, 1):
            try:
                status = action(item, by_id)
                error = ""
            except (HttpError, ValueError, RuntimeError, OSError) as exc:
                status = "ERROR"
                error = str(exc)
            record = {"id": item["id"], "path": item["path"], "status": status, "error": error}
            outcomes.append(record)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
            print(f"{index}/{len(ordered)} | {status} | {item['path']}")
            time.sleep(0.15)
        print("Kết quả:", dict(Counter(row["status"] for row in outcomes)))
        files.download(str(log_path))
        return outcomes
''')

markdown("""
    ## 03. Quét hoặc nạp danh sách

    **SEND:** tự quét và tải về `drive_transfer_manifest.json`. Giữ file này để dùng trên tài khoản mới.

    **ACCEPT:** tải lên đúng file JSON đã nhận ở lần `SEND`.
    Danh sách chỉ chứa ID, đường dẫn và thông tin file, không chứa checkpoint hay dữ liệu ảnh.
""")
code('''
    if MODE == "SEND":
        manifest = scan_folder()
        manifest_path = Path("drive_transfer_manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        files.download(str(manifest_path))
    else:
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Chỉ tải lên một file drive_transfer_manifest.json.")
        manifest = json.loads(next(iter(uploaded.values())).decode("utf-8"))

    by_id = validate_manifest(manifest)
    owned_bytes = sum(
        int(item.get("quotaBytesUsed", item.get("size", 0)))
        for item in manifest["items"]
        if owner_email(item) == SOURCE_EMAIL
    )
    print("Tổng file và thư mục:", len(manifest["items"]))
    print("Chủ sở hữu khi quét:", dict(Counter(owner_email(item) for item in manifest["items"])))
    print(f"Dung lượng thuộc tài khoản cũ khi quét: {owned_bytes / 1024**3:.2f} GiB")
    preview = pd.DataFrame([
        {"Đường dẫn": item["path"], "Chủ sở hữu": owner_email(item), "MiB": round(int(item.get("size", 0)) / 1024**2, 2)}
        for item in manifest["items"]
    ])
    display(preview)
    if MODE == "ACCEPT":
        quota = account.get("storageQuota", {})
        if "limit" in quota:
            available = int(quota["limit"]) - int(quota.get("usage", 0))
            print(f"Dung lượng trống tài khoản mới: {available / 1024**3:.2f} GiB")
            print("Số liệu khi quét có thể cao hơn thực tế nếu đã chuyển một phần.")
''')

markdown("""
    ## 04. Thực hiện

    Kiểm tra hai email, thư mục và danh sách bên trên.
    `SEND` nhập **GUI**; `ACCEPT` nhập **NHAN** để thực hiện.

    Nếu có `ERROR` hoặc `NOT_PENDING`, xem nhật ký. Có thể chạy lại cell này;
    file đã chuyển hoặc đã gửi yêu cầu được bỏ qua.
""")
code('''
    print("Từ:", SOURCE_EMAIL)
    print("Sang:", TARGET_EMAIL)
    print("Thư mục:", ROOT_FOLDER_URL)
    confirmation = "GUI" if MODE == "SEND" else "NHAN"
    if input(f"Nhập {confirmation} để tiếp tục: ").strip() != confirmation:
        raise RuntimeError("Đã dừng, chưa thực hiện chuyển quyền sở hữu.")
    outcomes = process_manifest(manifest)
''')

markdown("""
    ## 05. Kiểm tra chủ sở hữu và đưa thư mục vào Drive mới

    `SEND`: file còn thuộc tài khoản cũ là bình thường, vì tài khoản mới chưa nhận.

    `ACCEPT`: kiểm tra từng file một lần nữa. Chỉ đưa thư mục gốc vào **Drive của tôi**
    của tài khoản mới khi tất cả mục trong danh sách đã thuộc tài khoản mới.
    Cấu trúc thư mục con được giữ nguyên. File thuộc tài khoản thứ ba phải xử lý riêng.
""")
code('''
    audit = []
    for index, item in enumerate(manifest["items"], 1):
        try:
            current = check_live_item(item, by_id)
            owner = owner_email(current)
            status = "NEW_OWNER" if owner == TARGET_EMAIL else "OLD_OWNER" if owner == SOURCE_EMAIL else "OTHER_OWNER"
            error = ""
        except (HttpError, ValueError, RuntimeError, OSError) as exc:
            owner, status, error = "", "ERROR", str(exc)
        audit.append({"id": item["id"], "path": item["path"], "owner": owner, "status": status, "error": error})
        if index % 100 == 0:
            print("Đã kiểm tra:", index)

    audit_path = Path(f"drive_transfer_{MODE.lower()}_audit.csv")
    pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
    print("Chủ sở hữu hiện tại:", dict(Counter(row["status"] for row in audit)))
    remaining = [row for row in audit if row["status"] != "NEW_OWNER"]
    if remaining:
        display(pd.DataFrame(remaining))
    files.download(str(audit_path))

    if MODE == "ACCEPT" and not remaining:
        new_root = api_call(service.files().get(fileId="root", fields="id"))["id"]
        root = metadata(ROOT_ID)
        previous_parents = [parent for parent in root.get("parents", []) if parent != new_root]
        if new_root not in root.get("parents", []):
            options = {"fileId": ROOT_ID, "addParents": new_root, "fields": "id,name,parents"}
            if previous_parents:
                options["removeParents"] = ",".join(previous_parents)
            api_call(service.files().update(**options))
        final_root = metadata(ROOT_ID)
        if new_root not in final_root.get("parents", []):
            raise RuntimeError("Thư mục chưa nằm trong Drive của tôi của tài khoản mới.")
        print("Đã chuyển toàn bộ danh sách sang tài khoản mới.")
        print("Thư mục:", ROOT_FOLDER_URL)
        print("Tên thư mục:", final_root["name"])
    elif MODE == "ACCEPT":
        print("Chưa chuyển đủ. Xem danh sách còn lại; chưa thay đổi vị trí thư mục gốc.")
    else:
        print("Tiếp theo: mở notebook bằng tài khoản mới, chọn ACCEPT và tải lên JSON.")
''')

markdown("""
    ## 06. Chạy lại notebook training

    - Mở notebook training bằng tài khoản mới và kết nối Drive mới.
    - Nếu thư mục gốc vẫn tên `K4_Track4_Day2`, giữ đường dẫn hiện tại.
    - Giữ `FORCE_RERUN = False` để tiếp tục checkpoint trong `runs`.
    - Không cần xoá checkpoint cũ: đây là cùng file đã đổi chủ sở hữu.
    - File mới tạo bởi tài khoản cũ sau khi quét chưa nằm trong JSON. Dừng training trước khi chuyển;
      nếu phát sinh file mới, quét và chuyển bổ sung.

    **Nếu gặp lỗi:**

    - `insufficientPermissions`: mở phiên mới, đăng nhập đúng tài khoản và cấp quyền Drive cho Colab.
    - `storageQuotaExceeded`: tài khoản nhận cần thêm dung lượng, rồi chạy lại `ACCEPT`.
    - `NOT_PENDING`: chạy lại `SEND` bằng tài khoản cũ để gửi yêu cầu còn thiếu.
    - Không dùng chung một phiên Colab để đổi tài khoản giữa hai chế độ.

    Tài liệu: [Chuyển quyền sở hữu bằng API](https://developers.google.com/workspace/drive/api/guides/transfer-file),
    [Quyền sở hữu và dung lượng](https://support.google.com/drive/answer/2494892?hl=vi).
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "chuyen_quyen_drive_colab.ipynb"},
    },
    "cells": cells,
}
for index, cell in enumerate(cells):
    cell["id"] = f"drive-transfer-{index:02d}"
output = Path(__file__).resolve().parent / "chuyen_quyen_drive_colab.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(output)
